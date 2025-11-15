# analyze_classes_compare.py
import os
import re
import numpy as np
import pandas as pd

# RDKit is optional (better parsing if available)
try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

METALS = {
    'Li','Na','K','Rb','Cs','Fr','Be','Mg','Ca','Sr','Ba','Ra','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Al','Ga','In',
    'Tl','Sn','Pb','Bi','Sb'
}
HALOGENS = {'F','Cl','Br','I'}

# -----------------------------
# Helpers
# -----------------------------
def aggregate_predictions_per_cas(pred_csv, cas_col="CAS"):
    df = pd.read_csv(pred_csv)
    if cas_col not in df.columns:
        raise ValueError(f"'{cas_col}' not found in {pred_csv}")
    required = {"correct_w1b", "correct_exact"}
    if not required.issubset(df.columns):
        raise ValueError(f"{pred_csv} must contain {required}")
    df[cas_col] = df[cas_col].astype(str)
    agg = (df.groupby(cas_col, as_index=False)
             .agg(n_obs=("correct_w1b","size"),
                  n_correct_w1b=("correct_w1b","sum"),
                  n_correct_exact=("correct_exact","sum")))
    agg["p_w1b"]   = agg["n_correct_w1b"] / agg["n_obs"]
    agg["p_exact"] = agg["n_correct_exact"] / agg["n_obs"]
    return agg

def parse_smiles_flags(smi: str):
    """
    Returns a dict with:
      contains_metal (bool), n_metal (int)
      contains_halogen (bool), n_halogen (int)
      is_salt_mix (bool)  [dot-separated or >1 fragments]
      contains_formal_charge (bool), n_charged_atoms (int), total_abs_formal_charge (int)
      heavy_atoms (int), n_fragments (int)
    """
    out = dict(
        contains_metal=False, n_metal=0,
        contains_halogen=False, n_halogen=0,
        is_salt_mix=False,
        contains_formal_charge=False, n_charged_atoms=0, total_abs_formal_charge=0,
        heavy_atoms=np.nan, n_fragments=np.nan
    )
    if not isinstance(smi, str) or not smi:
        return out

    out["is_salt_mix"] = ("." in smi)

    if HAS_RDKIT:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            frags = Chem.GetMolFrags(m)
            out["n_fragments"] = len(frags)
            heavy = [a for a in m.GetAtoms() if a.GetAtomicNum() > 1]
            out["heavy_atoms"] = len(heavy)
            for a in heavy:
                sym = a.GetSymbol()
                if sym in METALS:
                    out["contains_metal"] = True
                    out["n_metal"] += 1
                if sym in HALOGENS:
                    out["contains_halogen"] = True
                    out["n_halogen"] += 1
                fc = a.GetFormalCharge()
                if fc != 0:
                    out["contains_formal_charge"] = True
                    out["n_charged_atoms"] += 1
                    out["total_abs_formal_charge"] += abs(int(fc))
            if (out["n_fragments"] is not np.nan) and out["n_fragments"] and out["n_fragments"] > 1:
                out["is_salt_mix"] = True
            return out


    n_hal = len(re.findall(r'(Cl|Br|I|F)', smi))
    out["n_halogen"] = n_hal
    out["contains_halogen"] = n_hal > 0

    out["contains_metal"] = any(tok in smi for tok in METALS)
    out["n_metal"] = 1 if out["contains_metal"] else 0

    # Charge heuristic
    n_charged = len(re.findall(r'\[.*?[\+\-].*?\]', smi))
    out["n_charged_atoms"] = n_charged
    out["contains_formal_charge"] = n_charged > 0

    heavy_guess = len(re.findall(r'[A-Z][a-z]?', smi))
    out["heavy_atoms"] = heavy_guess

    out["total_abs_formal_charge"] = n_charged  
    out["n_fragments"] = smi.count('.') + 1
    return out

def build_class_labels(flags_df):
    """
    Define boolean chemotype classes from flag columns.
    Returns a DataFrame with one boolean column per class.
    """
    f = flags_df.fillna(0)
    classes = pd.DataFrame(index=f.index)

    classes["halogenated"]      = f["contains_halogen"].astype(bool)
    classes["polyhalogenated"]  = (f["n_halogen"] >= 3)
    classes["heavy_metal"]      = f["contains_metal"].astype(bool)
    classes["charged"]          = f["contains_formal_charge"].astype(bool)
    classes["lots_of_ions"]     = (f["n_charged_atoms"] >= 2) | f["is_salt_mix"].astype(bool)
    classes["salt_mixture"]     = f["is_salt_mix"].astype(bool)

    ha = pd.to_numeric(f["heavy_atoms"], errors="coerce")
    classes["small"] = ha.lt(10)
    classes["large"] = ha.ge(20)
    classes["halogen_rich_large"] = classes["halogenated"] & classes["large"]

    classes["neutral_no_metal_no_halogen"] = (~classes["halogenated"]) & (~classes["heavy_metal"]) & (~classes["charged"])

    return classes

def summarize_by_classes(per_cas, classes, model_prefix):
    """
    per_cas: index=CAS; must contain p_w1b_<prefix>, p_exact_<prefix>, n_obs_<prefix>
    classes: boolean DataFrame indexed by CAS with class columns
    """
    # Resolve the right column names (fall back to unprefixed if needed)
    pw_col   = f"p_w1b_{model_prefix}"   if f"p_w1b_{model_prefix}"   in per_cas.columns else "p_w1b"
    pe_col   = f"p_exact_{model_prefix}" if f"p_exact_{model_prefix}" in per_cas.columns else "p_exact"
    nobs_col = f"n_obs_{model_prefix}"   if f"n_obs_{model_prefix}"   in per_cas.columns else "n_obs"

    missing = [c for c in (pw_col, pe_col) if c not in per_cas.columns]
    if missing:
        raise KeyError(f"summarize_by_classes: expected columns {missing} not found in per_cas")

    df = per_cas.join(classes, how="inner")
    rows = []
    for cname in classes.columns:
        mask = df[cname].fillna(False)
        sub = df[mask]
        if sub.empty:
            rows.append({
                "class": cname,
                f"n_compounds_{model_prefix}": 0,
                f"mean_p_w1b_{model_prefix}": np.nan,
                f"mean_p_exact_{model_prefix}": np.nan,
                "median_heavy": np.nan,
            })
            continue

        rows.append({
            "class": cname,
            f"n_compounds_{model_prefix}": int(sub.shape[0]),
            f"mean_p_w1b_{model_prefix}": float(sub[pw_col].mean()),
            f"mean_p_exact_{model_prefix}": float(sub[pe_col].mean()),
            "median_heavy": float(sub["heavy_atoms"].median()) if "heavy_atoms" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main analysis
# -----------------------------
def analyze_classes_compare(
    pred_csv_3d,
    pred_csv_gp,
    test_withfp_csv,
    cas_col="CAS",
    smiles_col="Canonical SMILES",
    out_dir="./analysis_outputs"
):
    os.makedirs(out_dir, exist_ok=True)

    agg3d = aggregate_predictions_per_cas(pred_csv_3d, cas_col=cas_col).rename(
        columns={"p_w1b":"p_w1b_3d","p_exact":"p_exact_3d","n_obs":"n_obs_3d"}
    )
    agggp = aggregate_predictions_per_cas(pred_csv_gp, cas_col=cas_col).rename(
        columns={"p_w1b":"p_w1b_gp","p_exact":"p_exact_gp","n_obs":"n_obs_gp"}
    )

    fp = pd.read_csv(test_withfp_csv)
    if cas_col not in fp.columns:
        raise ValueError(f"{cas_col} not in {test_withfp_csv}")
    if smiles_col not in fp.columns:
        # fallback
        if "SMILES" in fp.columns:
            smiles_col = "SMILES"
        else:
            raise ValueError(f"Neither '{smiles_col}' nor 'SMILES' found in {test_withfp_csv}")

    fp[cas_col] = fp[cas_col].astype(str)
    per_cas_smiles = (fp[[cas_col, smiles_col]]
                      .dropna(subset=[smiles_col])
                      .drop_duplicates(subset=[cas_col], keep="first")
                      .set_index(cas_col))

    # Compute flags
    flag_rows = [parse_smiles_flags(s) for s in per_cas_smiles[smiles_col].astype(str)]
    flags = pd.DataFrame(flag_rows, index=per_cas_smiles.index)

    # Build classes (boolean columns)
    classes = build_class_labels(flags)

    # Merge per-CAS aggregates with flags for convenient summaries
    per_cas_3d = (agg3d.set_index(cas_col)
                       .join(flags, how="left"))
    per_cas_gp = (agggp.set_index(cas_col)
                       .join(flags, how="left"))

    # Summaries per class for each model
    sum3d = summarize_by_classes(per_cas_3d, classes, model_prefix="3d")
    sumgp = summarize_by_classes(per_cas_gp, classes, model_prefix="gp")

    # Combine and compute deltas using only compounds present in BOTH models for each class
    # Build a per-class common-compounds mask and compute means on common set
    common_rows = []
    for cname in classes.columns:
        cidx = classes.index[classes[cname].fillna(False)]
        sub3d = per_cas_3d.loc[per_cas_3d.index.intersection(cidx)]
        subgp = per_cas_gp.loc[per_cas_gp.index.intersection(cidx)]
        common_ids = sub3d.index.intersection(subgp.index)
        if len(common_ids) == 0:
            common_rows.append({
                "class": cname, "n_common": 0,
                "mean_p_w1b_3d_common": np.nan, "mean_p_w1b_gp_common": np.nan, "delta_p_w1b": np.nan,
                "mean_p_exact_3d_common": np.nan, "mean_p_exact_gp_common": np.nan, "delta_p_exact": np.nan
            })
            continue
        m3_w1b = float(sub3d.loc[common_ids, "p_w1b_3d"].mean())
        mg_w1b = float(subgp.loc[common_ids, "p_w1b_gp"].mean())
        m3_ex  = float(sub3d.loc[common_ids, "p_exact_3d"].mean())
        mg_ex  = float(subgp.loc[common_ids, "p_exact_gp"].mean())
        common_rows.append({
            "class": cname, "n_common": int(len(common_ids)),
            "mean_p_w1b_3d_common": m3_w1b, "mean_p_w1b_gp_common": mg_w1b, "delta_p_w1b": m3_w1b - mg_w1b,
            "mean_p_exact_3d_common": m3_ex,  "mean_p_exact_gp_common":  mg_ex,  "delta_p_exact":  m3_ex  - mg_ex
        })
    common_df = pd.DataFrame(common_rows)

    summary = (sum3d.merge(sumgp, on="class", how="outer")
                    .merge(common_df, on="class", how="left"))

    # Save outputs
    out_summary = os.path.join(out_dir, "class_summary_compare.csv")
    out_percas  = os.path.join(out_dir, "class_compounds_compare.csv")

    summary.sort_values("class").to_csv(out_summary, index=False)
    
    per_cas_full = (per_cas_3d[["p_w1b_3d","p_exact_3d","n_obs_3d"]]
                    .join(per_cas_gp[["p_w1b_gp","p_exact_gp","n_obs_gp"]], how="outer")
                    .join(flags, how="left")
                    .join(classes, how="left"))
    per_cas_full.reset_index().rename(columns={per_cas_full.index.name or "index":"CAS"}).to_csv(out_percas, index=False)

    print(f"[INFO] Saved class summary:   {out_summary}")
    print(f"[INFO] Saved per-CAS detail:  {out_percas}")

def main():
    # EDIT THESE PATHS:
    pred_csv_3d = "./analysis_outputs/pred_3d_scaffold_regression_best.csv"
    pred_csv_gp = "./analysis_outputs/pred_filtered_mordred_scaffold_unknown_regression.csv"
    test_withfp_csv = "./molnet_dataset/dataset_scaffold_withfp_test.csv"

    analyze_classes_compare(
        pred_csv_3d=pred_csv_3d,
        pred_csv_gp=pred_csv_gp,
        test_withfp_csv=test_withfp_csv,
        cas_col="CAS",
        smiles_col="Canonical SMILES",
        out_dir="./analysis_outputs"
    )

if __name__ == "__main__":
    main()
