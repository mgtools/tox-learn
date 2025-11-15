#!/usr/bin/env python3
# cluster_stats_compare.py
# Find clusters where 3D model outperforms 2D (and vice versa), using the SAME clusters as in the plot.

import os, re, math, hashlib, random
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# =========================
# RNG hygiene
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# --- NEW HELPERS (place near your other utilities) -------------------
from rdkit import Chem

# Simple, interpretable SMARTS for common functional groups
FG_SMARTS = {
    "carboxylic_acid": Chem.MolFromSmarts("C(=O)[O;H1]"),
    "ester":           Chem.MolFromSmarts("C(=O)O[C;!$([#7,#8,#16])]"),
    "amide":           Chem.MolFromSmarts("C(=O)N"),
    "aldehyde":        Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    "ketone":          Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "alcohol":         Chem.MolFromSmarts("[CX4;!$(C=O)][OX2H]"),
    "phenol":          Chem.MolFromSmarts("c[OX2H]"),
    "amine_primary":   Chem.MolFromSmarts("[NX3;H2][CX4]"),
    "amine_secondary": Chem.MolFromSmarts("[NX3;H1]([CX4])[CX4]"),
    "amine_tertiary":  Chem.MolFromSmarts("[NX3]([CX4])([CX4])[CX4]"),
    "aniline":         Chem.MolFromSmarts("c[NX3]"),
    "nitrile":         Chem.MolFromSmarts("C#N"),
    "nitro":           Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    "ether":           Chem.MolFromSmarts("[CX4]-O-[CX4]"),
    "thioether":       Chem.MolFromSmarts("[#6]-S-[#6]"),
    "thiol":           Chem.MolFromSmarts("[#6][SH]"),
    "sulfonamide":     Chem.MolFromSmarts("S(=O)(=O)N"),
    "sulfonic_acid":   Chem.MolFromSmarts("S(=O)(=O)[O;H1]"),
    "halogen":         Chem.MolFromSmarts("[F,Cl,Br,I]"),
    "aromatic":        Chem.MolFromSmarts("a")
}

HALOGEN_ATOMIC_NUMS = {9, 17, 35, 53, 85}

def _pick_smiles_col(df: pd.DataFrame, fallback: str = None):
    candidates = ["Canonical SMILES", "SMILES", "Smiles"]
    if fallback is not None:
        candidates = [fallback] + [c for c in candidates if c != fallback]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No SMILES column found. Tried: {candidates}. Pass smiles_col=...")

def _mol_from_smiles(s: str):
    try:
        return Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    except Exception:
        return None

def _functional_group_hits(mol):
    """Return set of FG names present in mol (binary presence)."""
    if mol is None:
        return set()
    hits = set()
    for name, patt in FG_SMARTS.items():
        if patt is None: 
            continue
        if mol.HasSubstructMatch(patt):
            hits.add(name)
    return hits

def _halogen_fraction(mol):
    """(# halogen atoms) / (# heavy atoms). Returns np.nan if mol invalid or no heavy atoms."""
    if mol is None:
        return np.nan
    heavy = 0
    halos = 0
    for a in mol.GetAtoms():
        Z = a.GetAtomicNum()
        if Z != 1:  # not hydrogen
            heavy += 1
            if Z in HALOGEN_ATOMIC_NUMS:
                halos += 1
    return np.nan if heavy == 0 else halos / float(heavy)
# ----------------------------------------------------------------------

# =========================
# Utilities (copied/consistent with plotting script)
# =========================
def wilson_interval(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def numeric_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s))

def detect_fp_cols(df, start_index=20, non_fp_hints=None):
    if start_index is not None and start_index < len(df.columns):
        return list(df.columns[start_index:])
    if non_fp_hints is None: non_fp_hints = set()
    fp_cols = []
    for c in df.columns:
        if c in non_fp_hints: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > 1:
                fp_cols.append(c)
        else:
            try:
                sample = pd.to_numeric(df[c].head(100), errors="coerce")
                if sample.notna().sum() > 50: fp_cols.append(c)
            except Exception:
                pass
    return fp_cols

def coerce_numeric_frame(df):
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.notna().any(axis=0)]
    out = out.fillna(0.0)
    nunique = out.nunique(axis=0)
    out = out.loc[:, nunique > 1]
    return out

def make_clusterfp_kmeans(X_cont, k=64, tau=1.0, random_state=42):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_cont)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(Xz)
    D = pairwise_distances(Xz, km.cluster_centers_, metric="euclidean")  # [n,k]
    Z = -D / max(1e-6, float(tau))
    Z -= Z.max(axis=1, keepdims=True)
    S = np.exp(Z); S /= S.sum(axis=1, keepdims=True)   # soft assignment (for embedding)
    return S, scaler, km

def signature(ids, k, tau, seed):
    m = hashlib.sha1()
    m.update(("|".join(ids)).encode())
    m.update(f"|k={k}|tau={tau}|seed={seed}".encode())
    return m.hexdigest()[:12]

def build_alignment_ids(
    pred: pd.DataFrame,
    test_fp: pd.DataFrame,
    pred_key: str,
    test_key: str,
    id_map_csv: Optional[str] = None,
):
    # Optional explicit mapping first (strongest)
    join_col = pred_key
    if id_map_csv is not None:
        m = pd.read_csv(id_map_csv)
        if pred_key not in m.columns or test_key not in m.columns:
            raise KeyError(f"id_map_csv must have columns '{pred_key}' and '{test_key}'")
        m[pred_key] = m[pred_key].astype(str)
        m[test_key] = m[test_key].astype(str)
        pred = pred.merge(m[[pred_key, test_key]].drop_duplicates(), on=pred_key, how="left")
        join_col = test_key

    pred_ids = pred[join_col].astype(str)
    test_ids = test_fp[test_key].astype(str)

    overlap = set(pred_ids) & set(test_ids)
    if len(overlap) > 0:
        fp_cols = [c for c in test_fp.columns if c != test_key]
        raw_fp = (test_fp[[test_key] + fp_cols]
                  .drop_duplicates(test_key)
                  .set_index(test_key))
        fp_numeric = coerce_numeric_frame(raw_fp)
        ids_series = pred_ids
        ids_series = ids_series[ids_series.isin(fp_numeric.index.astype(str))]
        return ids_series, fp_numeric

    if pred_key == "title":
        pred = pred.copy()
        pred["_pred_numid"] = pred[pred_key].astype(str).str.extract(r"^(\d+)", expand=False).fillna("")
        test_fp = test_fp.copy()
        test_fp["_test_numid"] = test_fp[test_key].astype(str).map(numeric_only)
        if (pred["_pred_numid"] != "").sum() == 0:
            raise ValueError("Could not extract numeric prefix from any 'title'. Provide id_map_csv.")

        pred_ids = pred["_pred_numid"]
        test_ids = test_fp["_test_numid"]
        overlap = set(pred_ids) & set(test_ids)
        if len(overlap) == 0:
            sample_titles = pred[pred_key].head(5).tolist()
            raise ValueError(
                "No overlap after numeric-prefix mapping.\n"
                f"Example titles: {sample_titles}\n"
                f"Consider supplying id_map_csv with columns [{pred_key}, {test_key}]"
            )

        fp_cols = [c for c in test_fp.columns if c not in {test_key, "_test_numid"}]
        raw_fp = (test_fp[["_test_numid"] + fp_cols]
                  .drop_duplicates("_test_numid")
                  .set_index("_test_numid"))
        fp_numeric = coerce_numeric_frame(raw_fp)
        ids_series = pred_ids
        ids_series = ids_series[ids_series.isin(fp_numeric.index.astype(str))]
        return ids_series, fp_numeric

    raise ValueError(
        "No direct ID overlap. If keys differ, provide id_map_csv with columns "
        f"[{pred_key}, {test_key}]"
    )
def run_cluster_stats_compare(
    pred2d_csv,
    pred3d_csv,
    test_withfp_csv,
    compound_col="CAS",
    pred3d_key="title",
    test_key="CAS",
    id_map_csv: Optional[str] = None,
    fp_start_index=20,
    k_clusterfp=64,
    tau_clusterfp=1.0,
    seed=SEED,
    out_dir="./analysis_outputs",
    min_cluster_size=20,
    top_k=3,
):
    """
    Cluster-level comparison of 2D vs 3D.

    - Computes per-compound W1B and TRUE (exact) correctness for both models (if inputs allow).
    - Aggregates to clusters defined from the SAME fingerprint table as the plot.
    - SELECTS clusters using W1B delta (?W1B), but also reports TRUE accuracy side-by-side.
    - Outputs:
        cluster_stats__{A}__{B}.csv  (all clusters with W1B + TRUE metrics)
        cluster_selected__{A}__{B}.csv (top/bottom by ?W1B)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load
    pred2d = pd.read_csv(pred2d_csv)
    pred3d = pd.read_csv(pred3d_csv)
    test_fp = pd.read_csv(test_withfp_csv)

    # ---------- 2D: build correctness columns per sample ----------
    if compound_col not in pred2d.columns or compound_col not in test_fp.columns:
        raise KeyError(f"'{compound_col}' must be present in both pred2d and test files.")
    pred2d[compound_col] = pred2d[compound_col].astype(str)
    test_fp[test_key] = test_fp[test_key].astype(str)

    # Ensure EXACT for 2D; compute if needed
    if "correct_exact" not in pred2d.columns:
        if {"y_true_bin", "y_pred_bin"} <= set(pred2d.columns):
            pred2d["correct_exact"] = (pred2d["y_true_bin"].astype(int) == pred2d["y_pred_bin"].astype(int)).astype(int)
        else:
            raise KeyError("2D file lacks 'correct_exact' and cannot compute it (need y_true_bin & y_pred_bin).")
    # Ensure W1B for 2D; compute if needed
    if "correct_w1b" not in pred2d.columns:
        if {"y_true_bin", "y_pred_bin"} <= set(pred2d.columns):
            pred2d["correct_w1b"] = (np.abs(pred2d["y_true_bin"].astype(int) - pred2d["y_pred_bin"].astype(int)) <= 1).astype(int)
        else:
            raise KeyError("2D file lacks 'correct_w1b' and cannot compute it (need y_true_bin & y_pred_bin).")

    agg2d_spec = {
        "n2d": ("correct_exact", "size"),
        "k2d_exact": ("correct_exact", "sum"),
        "k2d_w1b": ("correct_w1b", "sum"),
    }
    agg2d = pred2d.groupby(compound_col, as_index=False).agg(**agg2d_spec)
    agg2d["p2d_exact"] = agg2d["k2d_exact"] / agg2d["n2d"]
    agg2d["p2d_w1b"]   = agg2d["k2d_w1b"] / agg2d["n2d"]

    # ---------- 3D: build correctness columns per sample ----------
    if {"y_true_bin", "pred_coral_bin"} <= set(pred3d.columns):
        pred3d[pred3d_key] = pred3d[pred3d_key].astype(str)
        pred3d["correct_exact"] = (pred3d["y_true_bin"].astype(int) == pred3d["pred_coral_bin"].astype(int)).astype(int)
        pred3d["correct_w1b"]   = (np.abs(pred3d["y_true_bin"].astype(int) - pred3d["pred_coral_bin"].astype(int)) <= 1).astype(int)
    else:
        raise KeyError("3D file must contain 'y_true_bin' and 'pred_coral_bin'.")

    ids_series, fp_numeric_full = build_alignment_ids(pred3d, test_fp, pred3d_key, test_key, id_map_csv=id_map_csv)
    pred3d_aligned = pred3d.copy()
    pred3d_aligned["_align_key"] = ids_series.values
    pred3d_aligned = pred3d_aligned[pred3d_aligned["_align_key"].notna()]

    agg3d = (pred3d_aligned.groupby("_align_key", as_index=False)
             .agg(n3d=("correct_exact", "size"),
                  k3d_exact=("correct_exact", "sum"),
                  k3d_w1b=("correct_w1b", "sum")))
    agg3d.rename(columns={"_align_key": compound_col}, inplace=True)
    agg3d[compound_col] = agg3d[compound_col].astype(str)
    agg3d["p3d_exact"] = agg3d["k3d_exact"] / agg3d["n3d"]
    agg3d["p3d_w1b"]   = agg3d["k3d_w1b"] / agg3d["n3d"]

    # ---------- IDs union to define FP space ----------
    ids_union = list(dict.fromkeys(
        list(agg2d[compound_col].astype(str).values) +
        list(agg3d[compound_col].astype(str).values)
    ))

    # Detect FP columns and build numeric FP table (same as plot)
    non_fp_hints = {compound_col, test_key, "Effect value", "Duration (hours)",
                    "y_true_bin", "y_pred_bin", "y_true_log10lc50", "y_pred_log10lc50",
                    "correct_w1b", "correct_exact", "Latin name", "_test_numid"}
    fp_cols = detect_fp_cols(test_fp, start_index=fp_start_index, non_fp_hints=non_fp_hints)
    if not fp_cols:
        raise ValueError("No fingerprint columns detected; check fp_start_index or dataset schema.")

    key_for_fp = compound_col if compound_col in test_fp.columns else test_key
    fps_indexed = (test_fp[[key_for_fp] + fp_cols]
                   .drop_duplicates(subset=[key_for_fp], keep="first")
                   .set_index(key_for_fp).astype(float))
    fps_numeric = coerce_numeric_frame(fps_indexed)

    ids_exist = [i for i in ids_union if i in fps_numeric.index]
    if not ids_exist:
        raise ValueError("After alignment, 0 IDs found in test fingerprints. Check keys/mapping.")

    X = fps_numeric.loc[ids_exist].values.astype(np.float64, copy=False)

    # ---------- SAME clusters as plot ----------
    S, scaler, km = make_clusterfp_kmeans(X, k=k_clusterfp, tau=tau_clusterfp, random_state=seed)
    labels = km.predict(scaler.transform(X))
    id_to_cluster = pd.Series(labels, index=ids_exist).to_dict()

    # ---------- Join per-ID correctness for intersection ----------
    joined = agg2d.merge(agg3d, on=compound_col, how="inner")
    if joined.empty:
        raise ValueError("No overlap between 2D and 3D aggregated IDs; cannot compare.")
    joined["cluster"] = joined[compound_col].map(id_to_cluster)
    joined = joined[joined["cluster"].notna()].copy()
    joined["cluster"] = joined["cluster"].astype(int)

    # ---------- Aggregate per cluster (W1B + TRUE) ----------
    agg_dict = dict(
        n2d=("n2d", "sum"),
        k2d_exact=("k2d_exact", "sum"),
        k2d_w1b=("k2d_w1b", "sum"),
        n3d=("n3d", "sum"),
        k3d_exact=("k3d_exact", "sum"),
        k3d_w1b=("k3d_w1b", "sum"),
        n_ids=("cluster", "size"),
    )
    grp = joined.groupby("cluster", as_index=False).agg(**agg_dict)

    # TRUE (exact) accuracy
    grp["p2d_exact"] = grp["k2d_exact"] / grp["n2d"]
    grp["p3d_exact"] = grp["k3d_exact"] / grp["n3d"]
    grp["delta_exact"] = grp["p3d_exact"] - grp["p2d_exact"]

    # W1B accuracy
    grp["p2d_w1b"] = grp["k2d_w1b"] / grp["n2d"]
    grp["p3d_w1b"] = grp["k3d_w1b"] / grp["n3d"]
    grp["delta_w1b"] = grp["p3d_w1b"] - grp["p2d_w1b"]

    # Wilson CIs (both metrics)
    p2e_lo, p2e_hi, p3e_lo, p3e_hi = [], [], [], []
    p2w_lo, p2w_hi, p3w_lo, p3w_hi = [], [], [], []
    for _, r in grp.iterrows():
        lo2e, hi2e = wilson_interval(int(r.k2d_exact), int(r.n2d))
        lo3e, hi3e = wilson_interval(int(r.k3d_exact), int(r.n3d))
        p2e_lo.append(lo2e); p2e_hi.append(hi2e)
        p3e_lo.append(lo3e); p3e_hi.append(hi3e)

        lo2w, hi2w = wilson_interval(int(r.k2d_w1b), int(r.n2d))
        lo3w, hi3w = wilson_interval(int(r.k3d_w1b), int(r.n3d))
        p2w_lo.append(lo2w); p2w_hi.append(hi2w)
        p3w_lo.append(lo3w); p3w_hi.append(hi3w)

    grp["p2d_exact_lo"], grp["p2d_exact_hi"] = p2e_lo, p2e_hi
    grp["p3d_exact_lo"], grp["p3d_exact_hi"] = p3e_lo, p3e_hi
    grp["p2d_w1b_lo"],  grp["p2d_w1b_hi"]  = p2w_lo, p2w_hi
    grp["p3d_w1b_lo"],  grp["p3d_w1b_hi"]  = p3w_lo, p3w_hi

    # Filter by size for robustness (by compounds)
    robust = grp[(grp["n_ids"] >= min_cluster_size)].copy()

    # ---------- Select top/bottom clusters by ?W1B (as requested) ----------
    top_pos = robust.sort_values("delta_w1b", ascending=False).head(top_k)
    top_neg = robust.sort_values("delta_w1b", ascending=True).head(top_k)

    # ---------- Save outputs ----------
    baseA = os.path.splitext(os.path.basename(pred2d_csv))[0]
    baseB = os.path.splitext(os.path.basename(pred3d_csv))[0]
    out_dir2 = os.path.join(out_dir, "cluster_stats")
    os.makedirs(out_dir2, exist_ok=True)

    all_csv = os.path.join(out_dir2, f"cluster_stats__{baseA}__{baseB}.csv")
    robust.to_csv(all_csv, index=False)

    sel_csv = os.path.join(out_dir2, f"cluster_selected__{baseA}__{baseB}.csv")
    pd.concat([
        top_pos.assign(direction="3D>>2D"),
        top_neg.assign(direction="2D>>3D")
    ], ignore_index=True).to_csv(sel_csv, index=False)

    # ---------- Print summary (show BOTH metrics) ----------
    def fmt_row(r):
        return (f"cluster={int(r.cluster):3d} | n_ids={int(r.n_ids):4d} | "
                f"W1B: 2D {r.p2d_w1b:.3f} [{r.p2d_w1b_lo:.3f},{r.p2d_w1b_hi:.3f}]  "
                f"vs 3D {r.p3d_w1b:.3f} [{r.p3d_w1b_lo:.3f},{r.p3d_w1b_hi:.3f}]  "
                f"?W1B={r.delta_w1b:+.3f}  ||  "
                f"EXACT: 2D {r.p2d_exact:.3f} [{r.p2d_exact_lo:.3f},{r.p2d_exact_hi:.3f}]  "
                f"vs 3D {r.p3d_exact:.3f} [{r.p3d_exact_lo:.3f},{r.p3d_exact_hi:.3f}]  "
                f"?EXACT={r.delta_exact:+.3f}")

    print("\n[Top clusters by ?W1B: 3D >> 2D (largest positive)]")
    for _, r in top_pos.iterrows():
        print("  " + fmt_row(r))

    print("\n[Top clusters by ?W1B: 2D >> 3D (largest negative)]")
    for _, r in top_neg.iterrows():
        print("  " + fmt_row(r))

    print(f"\n[OK] Wrote per-cluster stats (W1B + TRUE): {all_csv}")
    print(f"[OK] Wrote selected clusters (by ?W1B):    {sel_csv}")


# =========================
# Edit paths and run
# =========================
def main():
    # EDIT THESE to match the files you used for the figure:
    pred2d_csv = "./analysis_outputs/scaffold/pred__dataset__mordred__scaffold__s3__na__classification__GPBoost.csv"
    pred3d_csv = "./analysis_outputs/pred_3d_best_coral.csv"
    test_withfp_csv = "F:/molnet_dataset_nodot_flat/dataset_scaffold_s3_test_flat.csv"

    run_cluster_stats_compare(
        pred2d_csv=pred2d_csv,
        pred3d_csv=pred3d_csv,
        test_withfp_csv=test_withfp_csv,
        compound_col="CAS",
        pred3d_key="title",
        test_key="CAS",          # use "CAS" if that is your test key
        id_map_csv=None,         # provide a 2-col CSV if needed

        fp_start_index=20,
        k_clusterfp=64,
        tau_clusterfp=1.0,
        seed=SEED,
        out_dir="./analysis_outputs",
        min_cluster_size=20,     # adjust for your dataset size
        top_k=3,
    )

if __name__ == "__main__":
    main()
