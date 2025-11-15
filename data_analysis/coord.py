#!/usr/bin/env python3
# coord_overlay_autokey.py

import os, re, random
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pred2d_csv = r".\analysis_outputs\scaffold\pred__dataset__mordred__scaffold__s3__na__classification__GPBoost.csv"
pred3d_csv = r".\analysis_outputs\pred_3d_best_coral.csv"
test_withfp_csv = r"F:\molnet_dataset_nodot_flat\dataset_scaffold_s3_test_flat.csv"

id_map_csv: Optional[str] = r".\analysis_outputs\title_to_CAS_map_candidate.csv" 

selected_clusters_csv = r".\analysis_outputs\cluster_stats\cluster_selected__pred__dataset__mordred__scaffold__s3__na__classification__GPBoost__pred_3d_best_coral.csv"

fp_start_index = 20
k_clusterfp = 64
tau_clusterfp = 1.0
seed = 42

out_dir = r".\analysis_outputs"
out_png = os.path.join(out_dir, "cluster_overlay_selected__2d3d.png")


random.seed(seed)
np.random.seed(seed)


CAS_RE = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")

def strip_upper(s: str) -> str:
    return "" if s is None else str(s).strip().upper()

def numeric_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s)) if s is not None else ""

def normalize_cas(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    m = CAS_RE.search(s)
    if m: return m.group(1)
    digits = re.sub(r"\D+", "", s)
    if len(digits) >= 3:
        body = digits[:-3] or "0"; yy = digits[-3:-1]; z = digits[-1]
        try:
            return f"{int(body)}-{yy}-{z}"
        except Exception:
            pass
    return s.upper()

def normalize_title_numeric_prefix(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    m = re.match(r"^(\d+)", s)
    return m.group(1) if m else s.upper()

def detect_fp_cols(df: pd.DataFrame, start_index=20, non_fp_hints=None) -> List[str]:
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
                if sample.notna().sum() > 50:
                    fp_cols.append(c)
            except Exception:
                pass
    return fp_cols

def coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, out.notna().any(axis=0)].fillna(0.0)
    nunique = out.nunique(axis=0)
    out = out.loc[:, nunique > 1]
    return out

def make_clusterfp_kmeans(X_cont: np.ndarray, k=64, tau=1.0, random_state=42):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_cont)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(Xz)
    D = pairwise_distances(Xz, km.cluster_centers_, metric="euclidean")
    Z = -D / max(1e-6, float(tau))
    Z -= Z.max(axis=1, keepdims=True)
    S = np.exp(Z); S /= S.sum(axis=1, keepdims=True)
    return S, scaler, km

def embed_2d_auto(X: np.ndarray, seed=42, pca_dim=50):
    n, d = X.shape
    Xp = X
    if d > pca_dim:
        n_comp = min(pca_dim, max(2, n-1))
        Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2, init="pca", learning_rate=100,
        perplexity=60, n_iter=3000, early_exaggeration=8.0,
        angle=0.5, metric="euclidean", random_state=seed, verbose=0
    )
    return tsne.fit_transform(Xp)


def guess_test_id_column(test_df: pd.DataFrame, candidates: Optional[List[str]] = None) -> str:
    cand_cols = [c for c in test_df.columns
                 if test_df[c].dtype == object or pd.api.types.is_string_dtype(test_df[c])]
    cas_like = [c for c in cand_cols if "cas" in c.lower()]
    if cas_like:
        def score(col):
            s = test_df[col].astype(str)
            return s.str.contains(r"\d+-\d{2}-\d$", regex=True).sum()
        cas_like.sort(key=score, reverse=True)
        return cas_like[0]
    def uniq(col):
        s = test_df[col].astype(str)
        return s.nunique(dropna=True) / max(1, len(s))
    cand_cols.sort(key=uniq, reverse=True)
    return cand_cols[0] if cand_cols else test_df.columns[0]


def load_mapping_any_two_columns(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    M = pd.read_csv(path)
    if M.shape[1] < 2:
        raise ValueError(f"Mapping CSV must have at least two columns. Got {M.shape[1]}.")

    cols = list(M.columns)
    def title_score(col):
        s = str(col).lower()
        score = 0
        if "title" in s or "name" in s: score += 2
        val = M[col].astype(str)
        cas_hits = val.str.contains(r"\d+-\d{2}-\d$", regex=True, na=False).sum()
        score += max(0, 1000 - cas_hits) / 1000.0
        return score
    def cas_score(col):
        s = str(col).lower()
        score = 0
        if "cas" in s: score += 2
        val = M[col].astype(str)
        cas_hits = val.str.contains(r"\d+-\d{2}-\d$", regex=True, na=False).sum()
        score += cas_hits / max(1, len(val))
        return score
    title_col = max(cols, key=title_score)
    cas_col   = max(cols, key=cas_score)
    if title_col == cas_col and len(cols) >= 2:
        cols_sorted = sorted(cols, key=cas_score, reverse=True)
        cas_col = cols_sorted[1]

    M = M[[title_col, cas_col]].copy()
    M.columns = ["__map_title__", "__map_testid__"]

    M["__map_testid__"] = M["__map_testid__"].astype(str).map(normalize_cas)
    M["__map_title__"]  = M["__map_title__"].astype(str)
    return M


def main():
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    pred2d = pd.read_csv(pred2d_csv)
    pred3d = pd.read_csv(pred3d_csv)
    test_fp = pd.read_csv(test_withfp_csv)

    test_id_col = guess_test_id_column(test_fp)
    print(f"[INFO] Using test ID column: {test_id_col!r}")

    # Normalize core IDs
    if "CAS" in pred2d.columns:
        compound_col_2d = "CAS"
    else:
        cand = [c for c in pred2d.columns if pred2d[c].dtype == object]
        compound_col_2d = cand[0] if cand else pred2d.columns[0]
        print(f"[WARN] 'CAS' not in pred2d; using {compound_col_2d!r}")

    pred2d[compound_col_2d] = pred2d[compound_col_2d].astype(str)
    pred3d_title_col = None
    for k in ["title", "Title", "NAME", "name"]:
        if k in pred3d.columns:
            pred3d_title_col = k; break
    if pred3d_title_col is None:
        scols = [c for c in pred3d.columns if pred3d[c].dtype == object]
        pred3d_title_col = scols[0] if scols else pred3d.columns[0]
        print(f"[WARN] No 'title' column found in pred3d; using {pred3d_title_col!r}")
    pred3d[pred3d_title_col] = pred3d[pred3d_title_col].astype(str)

    test_fp[test_id_col] = test_fp[test_id_col].astype(str)

    # Prepare auto-normalized versions
    P2 = pred2d.copy()
    P3 = pred3d.copy()
    T  = test_fp.copy()

    P2["_id_raw"]  = P2[compound_col_2d].astype(str)
    P2["_id_norm"] = P2["_id_raw"].map(strip_upper)
    P2["_id_cas"]  = P2["_id_raw"].map(normalize_cas)

    P3["_t_raw"]   = P3[pred3d_title_col].astype(str)
    P3["_t_norm"]  = P3["_t_raw"].map(strip_upper)
    P3["_t_cas"]   = P3["_t_raw"].map(normalize_cas)
    P3["_t_num"]   = P3["_t_raw"].map(normalize_title_numeric_prefix)

    T ["_id_raw"]  = T [test_id_col].astype(str)
    T ["_id_norm"] = T ["_id_raw"].map(strip_upper)
    T ["_id_cas"]  = T ["_id_raw"].map(normalize_cas)

    M = load_mapping_any_two_columns(id_map_csv) if id_map_csv else None
    if M is not None:
        print(f"[INFO] Mapping CSV loaded with columns: {list(M.columns)}")

        P3 = P3.merge(M, left_on=pred3d_title_col, right_on="__map_title__", how="left")
        P3["_id_map"] = P3["__map_testid__"].astype(str)

    strategies = []
    def ov(p2, p3, t):
        return len(set(p2) & set(t)), len(set(p3) & set(t))

    strategies.append((
        "raw", P2["_id_raw"], P3["_t_raw"], T["_id_raw"], *ov(P2["_id_raw"], P3["_t_raw"], T["_id_raw"])
    ))
    strategies.append((
        "strip_upper", P2["_id_norm"], P3["_t_norm"], T["_id_norm"], *ov(P2["_id_norm"], P3["_t_norm"], T["_id_norm"])
    ))
    strategies.append((
        "normalize_cas", P2["_id_cas"], P3["_t_cas"], T["_id_cas"], *ov(P2["_id_cas"], P3["_t_cas"], T["_id_cas"])
    ))
    strategies.append((
        "3d_numeric_prefix_vs_cas", P2["_id_cas"], P3["_t_num"], T["_id_cas"], *ov(P2["_id_cas"], P3["_t_num"], T["_id_cas"])
    ))
    if M is not None:
        strategies.append((
            "explicit_mapping", P2["_id_cas"], P3["_id_map"], T["_id_cas"], *ov(P2["_id_cas"], P3["_id_map"], T["_id_cas"])
        ))

    diag = pd.DataFrame([{
        "strategy": s[0],
        "overlap_2d": s[4],
        "overlap_3d": s[5],
        "n2d_unique": len(set(s[1])),
        "n3d_unique": len(set(s[2])),
        "ntest_unique": len(set(s[3])),
    } for s in strategies])
    print("\n[ID overlap diagnostics]")
    print(diag.to_string(index=False))

    diag = diag.sort_values(by=["overlap_3d","overlap_2d"], ascending=False).reset_index(drop=True)
    chosen = diag.iloc[0]["strategy"]
    print(f"\n[Chosen strategy] {chosen}")

    pick = {row["strategy"]: row for row in [
        {"strategy": s[0], "p2": s[1], "p3": s[2], "t": s[3]} for s in strategies
    ]}[chosen]

    P2["_align_id"] = pick["p2"].astype(str).values
    P3["_align_id"] = pick["p3"].astype(str).values
    T ["_align_id"] = pick["t"].astype(str).values

    pd.DataFrame({"P2_align_sample": pd.Series(P2["_align_id"].unique()[:50])}).to_csv(os.path.join(out_dir, "_debug_align_P2.csv"), index=False)
    pd.DataFrame({"P3_align_sample": pd.Series(P3["_align_id"].unique()[:50])}).to_csv(os.path.join(out_dir, "_debug_align_P3.csv"), index=False)
    pd.DataFrame({"T_align_sample":  pd.Series(T["_align_id"].unique()[:50])}).to_csv(os.path.join(out_dir, "_debug_align_T.csv"),  index=False)

    ids_2d = P2["_align_id"].dropna().astype(str).unique().tolist()
    ids_3d = P3["_align_id"].dropna().astype(str).unique().tolist()
    ids_union = list(dict.fromkeys(ids_2d + ids_3d))

    fp_cols = detect_fp_cols(T, start_index=fp_start_index, non_fp_hints={test_id_col, "_align_id", "_id_raw", "_id_norm", "_id_cas"})
    if not fp_cols:
        raise ValueError("No fingerprint columns detected; check fp_start_index and schema.")
    fps_indexed = (T[["_align_id"] + fp_cols]
                   .drop_duplicates(subset=["_align_id"], keep="first")
                   .set_index("_align_id"))
    fps_numeric = coerce_numeric_frame(fps_indexed)

    ids_exist = [i for i in ids_union if i in fps_numeric.index]
    if not ids_exist:
        pd.Series(ids_union[:200]).to_csv(os.path.join(out_dir, "_ids_union_sample.csv"), index=False, header=["ids_union_sample"])
        pd.Series(fps_numeric.index[:200]).to_csv(os.path.join(out_dir, "_ids_in_test_fp_index.csv"), index=False, header=["ids_in_test_fp_index"])
        raise ValueError(
            "No IDs from union found in test fingerprints after mapping.\n"
            "Open _debug_align_*.csv and compare to _ids_in_test_fp_index.csv.\n"
            "If needed, fix your mapping CSV values to match the test ID column."
        )

    X = fps_numeric.loc[ids_exist].values.astype(np.float64, copy=False)

    S, scaler, km = make_clusterfp_kmeans(X, k=k_clusterfp, tau=tau_clusterfp, random_state=seed)
    labels = km.predict(scaler.transform(X))
    S_logit = np.log(np.clip(S, 1e-6, 1-1e-6) / np.clip(1 - S, 1e-6, 1-1e-6))
    Z = embed_2d_auto(S_logit, seed=seed)

    coords = pd.DataFrame({"id": ids_exist, "x": Z[:,0], "y": Z[:,1], "cluster": labels})

    def add_p_correct_from_preds(coords_df: pd.DataFrame, pred_csv: str, id_in_coords: str, id_in_pred: str, mode="w1b") -> pd.DataFrame:
        P = pd.read_csv(pred_csv)
        df = coords_df.copy()
        df[id_in_coords] = df[id_in_coords].astype(str)
        P[id_in_pred] = P[id_in_pred].astype(str)
        if "correct_w1b" not in P.columns or "correct_exact" not in P.columns:
            if {"y_true_bin", "pred_coral_bin"}.issubset(P.columns):
                P["correct_exact"] = (P["y_true_bin"].astype(int) == P["pred_coral_bin"].astype(int)).astype(int)
                P["correct_w1b"]   = (np.abs(P["y_true_bin"].astype(int) - P["pred_coral_bin"].astype(int)) <= 1).astype(int)
            elif {"y_true_bin", "y_pred_bin"}.issubset(P.columns):
                P["correct_exact"] = (P["y_true_bin"].astype(int) == P["y_pred_bin"].astype(int)).astype(int)
                P["correct_w1b"]   = (np.abs(P["y_true_bin"].astype(int) - P["y_pred_bin"].astype(int)) <= 1).astype(int)
            else:
                raise KeyError("Cannot find/derive correctness columns in pred_csv.")
        correct_col = "correct_w1b" if mode.lower() == "w1b" else "correct_exact"
        agg = (P.groupby(id_in_pred, as_index=True)[correct_col].agg(n="size", k="sum"))
        agg["p_correct"] = agg["k"] / agg["n"]
        return df.merge(agg[["p_correct"]], left_on=id_in_coords, right_index=True, how="left")

    A = add_p_correct_from_preds(
        coords_df=coords.rename(columns={"id": "_align_id"}),
        pred_csv=pred2d_csv,
        id_in_coords="_align_id",
        id_in_pred=compound_col_2d,
        mode="w1b"
    ).rename(columns={"_align_id": "id"})

    tmp3d = P3[[pred3d_title_col, "_align_id"]].merge(pred3d, on=pred3d_title_col, how="right")
    tmp3d_path = os.path.join(out_dir, "_tmp3d_for_pcorrect.csv")
    tmp3d.to_csv(tmp3d_path, index=False)

    B = add_p_correct_from_preds(
        coords_df=coords.rename(columns={"id": "_align_id"}),
        pred_csv=tmp3d_path,
        id_in_coords="_align_id",
        id_in_pred="_align_id",
        mode="w1b"
    ).rename(columns={"_align_id": "id"})

    try: os.remove(tmp3d_path)
    except Exception: pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap("RdYlGn")

    A_plot = A.dropna(subset=["p_correct"]).copy()
    s1 = ax1.scatter(A_plot["x"], A_plot["y"], c=A_plot["p_correct"],
                     cmap=cmap, vmin=vmin, vmax=vmax, s=30, alpha=0.9,
                     linewidths=0.3, edgecolors="k")
    ax1.set_title("2D model - per-ID correctness (W1B)")
    ax1.set_xlabel("Embedding-1")
    ax1.set_ylabel("Embedding-2")

    B_plot = B.dropna(subset=["p_correct"]).copy()
    s2 = ax2.scatter(B_plot["x"], B_plot["y"], c=B_plot["p_correct"],
                     cmap=cmap, vmin=vmin, vmax=vmax, s=30, alpha=0.9,
                     linewidths=0.3, edgecolors="k")
    ax2.set_title("3D model - per-ID correctness (W1B)")
    ax2.set_xlabel("Embedding-1")
    ax2.set_ylabel("")

    cbar = fig.colorbar(s2, ax=[ax1, ax2], shrink=0.85, pad=0.02)
    cbar.set_label("Correctness rate (p_correct)")

    if os.path.exists(selected_clusters_csv):
        sel = pd.read_csv(selected_clusters_csv)
        if not sel.empty:
            sel["cluster"] = sel["cluster"].astype(int)
            highlight = sel["cluster"].unique().tolist()
            centroids = (coords[coords["cluster"].isin(highlight)]
                         .groupby("cluster", as_index=True)[["x","y"]].mean())
            for cid in highlight:
                if cid not in centroids.index: continue
                x_mean, y_mean = centroids.loc[cid, ["x","y"]].values
                direction = sel.loc[sel["cluster"] == cid, "direction"].iloc[0]
                color = "blue" if "3D>>2D" in direction else "red"
                for ax in (ax1, ax2):
                    ax.text(x_mean, y_mean, str(cid),
                            fontsize=13, weight="bold", color=color,
                            ha="center", va="center",
                            bbox=dict(facecolor="white", alpha=0.75, boxstyle="circle,pad=0.2"))
    else:
        print(f"[WARN] No selected_clusters_csv found: {selected_clusters_csv}")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved overlay: {out_png}")

if __name__ == "__main__":
    main()
