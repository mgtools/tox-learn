
import os
import re
import argparse
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

try:
    from rdkit import Chem
    RDKit_OK = True
except Exception:
    RDKit_OK = False


def bh_fdr(pvals):

    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    min_val = 1.0
    for i in range(n - 1, -1, -1):
        val = (n / float(i + 1)) * ranked[i]
        if val < min_val:
            min_val = val
        q[i] = min_val
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out
def aggregate_per_compound(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple test occurrences per compound to one row (equal weight per CAS).
    We average correctness across occurrences and keep SMILES/first seen structure.
    """
    keep_first = lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan
    per_cmpd = (per_sample_df
        .groupby("CAS", as_index=False)
        .agg(
            n_obs=("y_true_bin", "size"),
            SMILES=("SMILES", keep_first),
            model2d_exact=("model2d_exact", "mean"),
            model3d_exact=("model3d_exact", "mean"),
            model2d_w1b=("model2d_w1b", "mean"),
            model3d_w1b=("model3d_w1b", "mean")
        )
    )
    return per_cmpd


def paired_bootstrap_delta(a_correct, b_correct, n_boot=5000, seed=42):

    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def numeric_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s))

def within_one_bin_correct(y_true_bin, y_pred_bin):
    return (np.abs(np.asarray(y_pred_bin, int) - np.asarray(y_true_bin, int)) <= 1).astype(int)

def w1b_accuracy(y_true_bin, y_pred_bin) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred_bin, int) - np.asarray(y_true_bin, int)) <= 1))

def paired_binary_stats(a_correct: np.ndarray, b_correct: np.ndarray):

    a = a_correct.astype(int)
    b = b_correct.astype(int)
    delta = float(b.mean() - a.mean())

    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    table = [[int(np.sum((a == 0) & (b == 0))), n01],
             [n10, int(np.sum((a == 1) & (b == 1)))]]
    try:
        exact = (n01 + n10) <= 25
        res = mcnemar(table, exact=exact, correction=not exact)
        mcnemar_p = float(res.pvalue)
    except Exception:
        mcnemar_p = np.nan

    diffs = (b - a).astype(int)
    try:
        if np.all(diffs == 0):
            wilcoxon_p = 1.0
        else:
            wilcoxon_p = float(stats.wilcoxon(diffs, zero_method="wilcox", correction=True).pvalue)
    except Exception:
        wilcoxon_p = np.nan

    return delta, mcnemar_p, wilcoxon_p

def align_pred_files(pred2d, pred3d, test_df,
                     pred2d_key="CAS",
                     pred3d_key="title",
                     test_key="CAS",
                     smiles_col="Canonical SMILES"):
    for df, key in [(pred2d, pred2d_key), (pred3d, pred3d_key), (test_df, test_key)]:
        if key not in df.columns:
            raise KeyError(f"Key '{key}' missing in a dataframe.")
        df[key] = df[key].astype(str)

    pred3d = pred3d.copy()
    test_df = test_df.copy()
    if pred3d_key == "title":
        pred3d["_numkey"] = pred3d[pred3d_key].str.extract(r"^(\d+)", expand=False).fillna("")
        test_df["_numkey"] = test_df[test_key].astype(str).map(numeric_only)
        overlap = set(pred3d["_numkey"]) & set(test_df["_numkey"])
        left_key, right_key = (("_numkey","_numkey") if len(overlap)>0 else (pred3d_key, test_key))
    else:
        left_key, right_key = pred3d_key, test_key

    keep_cols = [c for c in [test_key, smiles_col, "Effect value"] if c in test_df.columns]
    base = test_df[keep_cols].copy()
    base = base.rename(columns={test_key: "CAS"})
    if smiles_col in base.columns:
        base = base.rename(columns={smiles_col: "SMILES"})
    if "Effect value" not in base.columns:
        raise KeyError("'Effect value' missing in test file (needed for ground truth).")

    base["y_true_log10lc50"] = np.log10(pd.to_numeric(base["Effect value"], errors="coerce"))
    bins = np.digitize(base["y_true_log10lc50"].values, [-1, 0, 1, 2], right=False)
    base["y_true_bin"] = bins.astype(int)

    p2 = pred2d.copy().rename(columns={pred2d_key: "CAS"})
    req2 = {"y_pred_bin", "correct_exact", "correct_w1b"}
    if not req2.issubset(p2.columns):
        missing = ", ".join(sorted(req2 - set(p2.columns)))
        raise KeyError(f"pred2d_csv missing columns: {missing}")
    p2 = p2[["CAS", "y_pred_bin", "correct_exact", "correct_w1b"]]
    p2 = p2.rename(columns={"y_pred_bin":"y_pred_bin_2d",
                            "correct_exact":"model2d_exact",
                            "correct_w1b":"model2d_w1b"})

    p3 = pred3d.copy()
    if left_key != "CAS":
        temp = pred3d[[pred3d_key, left_key]].drop_duplicates()
        cas_map = test_df[[test_key, right_key]].drop_duplicates()
        cas_map = cas_map.rename(columns={test_key:"CAS", right_key:left_key})
        p3 = temp.merge(cas_map, on=left_key, how="left").merge(pred3d, on=pred3d_key, how="left")
    else:
        p3 = p3.rename(columns={left_key:"CAS"})
    if "pred_coral_bin" not in p3.columns:
        raise KeyError("pred3d_csv must contain 'pred_coral_bin'.")
    p3 = p3.rename(columns={"pred_coral_bin":"y_pred_bin_3d"})
    p3 = p3[["CAS","y_pred_bin_3d"]]  

    merged = base.merge(p2, on="CAS", how="inner").merge(p3, on="CAS", how="inner")

    merged["model2d_exact"] = (merged["y_pred_bin_2d"].astype(int) == merged["y_true_bin"].astype(int)).astype(int)
    merged["model3d_exact"] = (merged["y_pred_bin_3d"].astype(int) == merged["y_true_bin"].astype(int)).astype(int)
    merged["model2d_w1b"]   = within_one_bin_correct(merged["y_true_bin"], merged["y_pred_bin_2d"])
    merged["model3d_w1b"]   = within_one_bin_correct(merged["y_true_bin"], merged["y_pred_bin_3d"])

    return merged  


def neutralize_nitro(mol):
    from rdkit import Chem
    patt = Chem.MolFromSmarts("[N+](=O)[O-]")
    repl = Chem.MolFromSmiles("N(=O)=O")
    try:
        if mol is None:
            return None
        rms = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=True)
        return rms[0] if rms else mol
    except Exception:
        return mol

def make_functional_group_flags(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    dedup_on: str = "CAS",          
    count_metals: bool = False,   
    neutralize_nitro_flag: bool = True
) -> pd.DataFrame:

    df = df.copy()
    if dedup_on in df.columns:
        df = df.drop_duplicates(subset=[dedup_on]).reset_index(drop=True)
    elif "CAS" in df.columns:
        df = df.drop_duplicates(subset=["CAS"]).reset_index(drop=True)

    if smiles_col not in df.columns:
        df["heavy_atoms"] = np.nan
        df["Halogen"] = 0
        df["Charged"] = 0
        df["Metal"] = 0
        return df

    if RDKit_OK:
        from rdkit import Chem
        mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in df[smiles_col]]
        if neutralize_nitro_flag:
            mols = [neutralize_nitro(m) for m in mols]

        heavy = []
        charged = []
        halogen = []
        metal = []

        metal_atomic_nums = set() if not count_metals else {
            3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,
            40,41,42,43,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,
            78,79,80,81
        }

        for m in mols:
            if m is None:
                heavy.append(np.nan); charged.append(0); metal.append(0); halogen.append(0)
                continue
            heavy.append(m.GetNumHeavyAtoms())
            has_charge = any(a.GetFormalCharge() != 0 for a in m.GetAtoms())
            charged.append(1 if has_charge else 0)
            has_metal = any(a.GetAtomicNum() in metal_atomic_nums for a in m.GetAtoms())
            metal.append(1 if has_metal else 0)
            has_hal = any(a.GetSymbol() in {"F","Cl","Br","I"} for a in m.GetAtoms())
            halogen.append(1 if has_hal else 0)
        df["heavy_atoms"] = heavy
        df["Charged"] = charged
        df["Metal"] = metal if count_metals else 0
        df["Halogen"] = halogen

        smarts = {
            "Aromatic": "a",
            "Hydroxyl": "[OX2H]",
            "Carbonyl": "[CX3]=O",
            "Amine": "[NX3;H2,H1;!$(NC=O)]",
            "Amide": "C(=O)N",
            "Carboxyl": "C(=O)[OX1H0-,OX2H1]",
            "Ester": "C(=O)O",
            "Ether": "[$([OX2]([#6])[#6])]",
            "Thiol": "[SH]",
            "Thioether": "[#16X2]([#6])[#6]",
            "Sulfonyl": "S(=O)(=O)",
            "HeteroAromatic": "[a;!c]",
            "Phenyl": "c1ccccc1",
            "Nitro": "N(=O)=O",  
        }
        patt = {k: Chem.MolFromSmarts(v) for k, v in smarts.items()}
        for name, p in patt.items():
            df[name] = [1 if (m and m.HasSubstructMatch(p)) else 0 for m in mols]
    else:
        df["heavy_atoms"] = np.nan
        df["Halogen"] = 0
        df["Charged"] = 0
        df["Metal"] = 0

    return df


def summarize_accuracy_by_functional_group(df_cmpd: pd.DataFrame, out_csv: str):
    """
    df_cmpd: one row per CAS, with mean correctness (0..1) columns.
    """
    core_exclude = {
        "CAS","SMILES","n_obs",
        "model2d_exact","model3d_exact","model2d_w1b","model3d_w1b",
        "heavy_atoms","Halogen","Charged","Metal"
    }
    group_cols = [c for c in df_cmpd.columns
                  if c not in core_exclude and df_cmpd[c].dropna().isin([0,1]).all()]

    recs = []
    for col in group_cols:
        sub = df_cmpd[df_cmpd[col] == 1]
        if len(sub) == 0:
            continue
        recs.append({
            "Functional Group": col,
            "N": int(len(sub)),
            "Median Heavy Atoms": float(np.nanmedian(sub.get("heavy_atoms", np.nan))),
            "Halogen Fraction": float(np.nanmean(sub.get("Halogen", np.nan))),
            "Accuracy_2D": float(sub["model2d_exact"].mean()),
            "Accuracy_3D": float(sub["model3d_exact"].mean()),
            "W1B_2D": float(sub["model2d_w1b"].mean()),
            "W1B_3D": float(sub["model3d_w1b"].mean()),
        })
    out = pd.DataFrame(recs)
    if not out.empty:
        out["?Acc"] = out["Accuracy_3D"] - out["Accuracy_2D"]
        out["?W1B"] = out["W1B_3D"] - out["W1B_2D"]
        out = out.sort_values("?W1B", ascending=False)
        out.to_csv(out_csv, index=False)
    return out


def main():
    pred2d_csv    = "./analysis_outputs/scaffold/pred__dataset__mordred__scaffold__s3__na__classification__GPBoost.csv"
    pred3d_csv    = "./analysis_outputs/pred_3d_best_coral.csv"
    test_withfp   = "F:/molnet_dataset_nodot_flat/dataset_scaffold_s3_test_flat.csv"

    pred2d = pd.read_csv(pred2d_csv)
    pred3d = pd.read_csv(pred3d_csv)
    testdf = pd.read_csv(test_withfp)

    per_sample = align_pred_files(pred2d, pred3d, testdf,
                                  pred2d_key="CAS",
                                  pred3d_key="title",
                                  test_key="CAS",
                                  smiles_col="Canonical SMILES")
    per_sample.to_csv("./analysis_outputs/merged_per_sample_correctness.csv", index=False)

    per_cmpd = aggregate_per_compound(per_sample)
    per_cmpd.to_csv("./analysis_outputs/merged_per_compound_correctness.csv", index=False)

    per_cmpd_with_flags = make_functional_group_flags(per_cmpd, smiles_col="SMILES", dedup_on="CAS",
                                                      count_metals=False, neutralize_nitro_flag=True)

    summary = summarize_accuracy_by_functional_group(per_cmpd_with_flags,
                                                     "./analysis_outputs/functional_group_acc_summary.csv")


    d_micro, p_mcn_micro, p_wil_micro = paired_binary_stats(
        per_sample["model2d_w1b"].values.astype(int),
        per_sample["model3d_w1b"].values.astype(int)
    )
    a = (per_cmpd["model2d_w1b"] >= 0.5).astype(int).values
    b = (per_cmpd["model3d_w1b"] >= 0.5).astype(int).values
    d_macro, p_mcn_macro, p_wil_macro = paired_binary_stats(a, b)

    print(f"[Paired stats] micro ?={d_micro:.3f}, McNemar p={p_mcn_micro:.3g}, Wilcoxon p={p_wil_micro:.3g}")
    print(f"[Paired stats] macro ?={d_macro:.3f}, McNemar p={p_mcn_macro:.3g}, Wilcoxon p={p_wil_macro:.3g}")

    lo, hi = paired_bootstrap_delta(a, b, n_boot=5000, seed=42)
    print(f"[Bootstrap ? W1B (macro)] 95% CI = [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()



