#!/usr/bin/env python3
"""
Overall chemical-type statistics for tox_pred paper:
 - Annotates both train and test splits (group split)
 - Deduplicates optionally by unique SMILES or unique CAS
 - Detects elements, metals, functional groups (incl. aromatic), and scaffolds
 - Produces tables and publication-quality pie plots
"""

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as e:
    raise SystemExit(f"RDKit is required. Install via `pip install rdkit-pypi`.\nError: {e}")

TRAIN_CSV = "./splits/groupsplit_train.csv"
TEST_CSV  = "./splits/groupsplit_test.csv"
SMILES_COL = "Canonical SMILES"
OUTDIR = Path("./analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

DO_UNIQUE_BY_SMILES = True
DO_UNIQUE_BY_CAS = True

GRAPHVIZ_VIVID = [
    "#E24A33","#348ABD","#988ED5","#777777","#FBC15E","#8EBA42","#FF9DA7",
    "#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD","#1F77B4",
    "#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F",
    "#BCBD22","#17BECF",
]

CAS_COL_CANDIDATES = ["CAS", "CAS Number", "CAS_Number", "cas", "cas_number"]

def pick_cas_col(df: pd.DataFrame):
    for c in CAS_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No CAS column found. Tried: {CAS_COL_CANDIDATES}")

def representative_smiles_per_cas(df: pd.DataFrame, cas_col: str, smiles_col: str):
    """Return representative SMILES per CAS (most frequent SMILES per CAS)."""
    tmp = df[[cas_col, smiles_col]].dropna(subset=[cas_col]).copy()
    counts = (
        tmp.groupby([cas_col, smiles_col], dropna=False)
           .size().reset_index(name="n")
           .sort_values([cas_col, "n"], ascending=[True, False])
    )
    rep = counts.drop_duplicates(subset=[cas_col], keep="first")[[cas_col, smiles_col]]

    missing = rep[smiles_col].isna()
    if missing.any():
        fill = (
            tmp[tmp[smiles_col].notna()]
            .drop_duplicates(subset=[cas_col], keep="first")
            [[cas_col, smiles_col]]
        )
        rep = rep.dropna(subset=[smiles_col]).merge(
            fill, on=cas_col, how="outer", suffixes=("", "_fill")
        )
        rep[smiles_col] = rep[smiles_col].fillna(rep.pop(f"{smiles_col}_fill"))
    return rep


SMARTS = {
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
SMARTS_PATTS = SMARTS 


METAL_SYMBOLS = {
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi","Po",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
}
def is_metal_symbol(sym: str) -> bool:
    return sym in METAL_SYMBOLS


def parse_smiles(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
        mol = frags[0]
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol

def element_counts(mol):
    counts, metals = Counter(), Counter()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] += 1
        if is_metal_symbol(sym):
            metals[sym] += 1
    return dict(counts), dict(metals)

def classify_type(mol, elem_counts, metal_counts):
    has_C = elem_counts.get("C", 0) > 0
    has_metal = sum(metal_counts.values()) > 0
    tags = []
    tags.append("single_component")
    if has_C and not has_metal:
        tags.append("organic")
    elif has_C and has_metal:
        tags.append("organometallic")
    elif (not has_C) and has_metal:
        tags.append("inorganic_metal")
    else:
        tags.append("inorganic_nonmetal")
    halos = sum(elem_counts.get(x, 0) for x in ["F", "Cl", "Br", "I"])
    if halos > 0:
        tags.append("halogenated")
    return tags

def functional_group_hits(mol):
    hits = {}
    for name, patt in SMARTS_PATTS.items():
        hits[name] = bool(patt and mol.HasSubstructMatch(patt))
    return hits

def basic_structure_stats(mol):
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    n_arom_bonds = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    n_bonds = mol.GetNumBonds()
    arom_frac = (n_arom_bonds / n_bonds) if n_bonds else 0.0
    return {
        "n_rings": n_rings,
        "aromatic_bond_fraction": round(arom_frac, 4),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "logP": round(Descriptors.MolLogP(mol), 3),
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
    }

def murcko_scaffold(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def annotate_dataframe(df, label="overall"):
    df = df.copy()
    df["__row_id__"] = np.arange(len(df))
    annots = []

    for _, row in df.iterrows():
        smi = row.get(SMILES_COL, None)
        mol = parse_smiles(smi)
        rec = {"__row_id__": row["__row_id__"], "parsed": mol is not None}
        if mol is None:
            annots.append(rec)
            continue
        elems, metals = element_counts(mol)
        tags = classify_type(mol, elems, metals)
        fg = functional_group_hits(mol)
        st = basic_structure_stats(mol)
        scaff = murcko_scaffold(mol)
        rec.update({
            "element_counts_json": json.dumps(elems),
            "metal_counts_json": json.dumps(metals),
            "type_tags_json": json.dumps(tags),
            "scaffold": scaff,
            **{f"FG_{k}": bool(v) for k, v in fg.items()},
            **st,
        })
        annots.append(rec)

    ann_df = pd.DataFrame(annots)
    out = df.merge(ann_df, on="__row_id__", how="left")

    # unpack selected elements
    def unpack_counts(col, element):
        return out[col].fillna("{}").apply(
            lambda x: json.loads(x).get(element, 0) if isinstance(x, str) else 0
        )

    for e in ["C","H","N","O","S","P","F","Cl","Br","I","Si","B","Na","K","Ca","Mg","Zn","Cu","Fe","Hg","Al","Sn","Pb"]:
        out[f"elem_{e}"] = unpack_counts("element_counts_json", e)

    out["has_metal"] = out["metal_counts_json"].fillna("{}").apply(
        lambda s: (sum(json.loads(s).values()) if isinstance(s, str) else 0) > 0
    )
    out["is_halogenated"] = (out[["elem_F","elem_Cl","elem_Br","elem_I"]].sum(axis=1) > 0)

    def first_label(js):
        try:
            arr = json.loads(js) if isinstance(js, str) else []
            for key in ["organometallic", "organic", "inorganic_metal", "inorganic_nonmetal"]:
                if key in arr:
                    return key
            return arr[0] if arr else "unknown"
        except Exception:
            return "unknown"

    out["coarse_type"] = out["type_tags_json"].apply(first_label)
    out_path = OUTDIR / f"annotated_{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"Annotated {label} saved -> {out_path}")
    return out


def save_fancy_pie(labels, values, title, outfile_prefix, max_slices=12, palette=GRAPHVIZ_VIVID):
    from matplotlib.patches import Circle
    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    if len(pairs) > max_slices:
        head = pairs[:max_slices-1]
        tail = pairs[max_slices-1:]
        other_sum = sum(v for _, v in tail)
        pairs = head + [("Other", other_sum)]
    labels, values = zip(*pairs)
    total = sum(values)
    fracs = [v/total for v in values]
    colors = (palette * ((len(labels)//len(palette))+1))[:len(labels)]
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, colors=colors,
                       wedgeprops=dict(linewidth=1.5, edgecolor="white"))
    ax.add_artist(Circle((0,0), 0.62, fc="white", ec="white", lw=1.5))
    ax.set_title(title, fontsize=12, pad=16)
    fmt = [f"{lab}  {val:,} ({frac*100:.1f}%)" for lab, val, frac in zip(labels, values, fracs)]
    ax.legend(wedges, fmt, loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.savefig(OUTDIR / f"{outfile_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTDIR / f"{outfile_prefix}.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved pie: {outfile_prefix}.png/.svg")


def summarize(annot, label):
    type_counts = annot["coarse_type"].value_counts(dropna=False).rename_axis("coarse_type").reset_index(name="count")
    type_counts["fraction"] = type_counts["count"] / type_counts["count"].sum()

    elem_cols = [c for c in annot.columns if c.startswith("elem_")]
    elem_presence = (annot[elem_cols] > 0).sum().sort_values(ascending=False).rename_axis("element").reset_index(name="n_compounds")

    fg_cols = [c for c in annot.columns if c.startswith("FG_")]
    fg_counts = (annot[fg_cols].sum().sort_values(ascending=False)
                 .rename_axis("functional_group").reset_index(name="n_compounds")
                 if fg_cols else pd.DataFrame(columns=["functional_group","n_compounds"]))
    fg_counts["functional_group"] = fg_counts["functional_group"].str.replace("FG_", "", regex=False)

    # Save summaries
    type_counts.to_csv(OUTDIR / f"overall_type_{label}.csv", index=False)
    elem_presence.to_csv(OUTDIR / f"overall_elements_{label}.csv", index=False)
    fg_counts.to_csv(OUTDIR / f"overall_functional_groups_{label}.csv", index=False)

    # Pies
    save_fancy_pie(type_counts["coarse_type"].tolist(), type_counts["count"].tolist(),
                   f"Coarse Chemical Types ({label})", f"overall_pie_coarse_type_{label}")
    # Elements pie
    top_elems = elem_presence.head(15)
    labels = top_elems["element"].tolist()
    values = top_elems["n_compounds"].tolist()
    if len(elem_presence) > 15:
        labels.append("Other")
        values.append(int(elem_presence["n_compounds"][15:].sum()))
    save_fancy_pie(labels, values, f"Elements Present ({label})", f"overall_pie_elements_{label}")

    if not fg_counts.empty:
        top_fg = fg_counts.head(15)
        save_fancy_pie(top_fg["functional_group"].tolist(), top_fg["n_compounds"].tolist(),
                       f"Functional Groups ({label})", f"overall_pie_functional_groups_{label}")

    return {"type_counts": type_counts, "elem_presence": elem_presence, "fg_counts": fg_counts}


def main():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    both = pd.concat([train, test], ignore_index=True)
    print(f"Loaded: train={train.shape}, test={test.shape}, combined={both.shape}")

    # all rows
    ann_all = annotate_dataframe(both, "overall_allrows")
    summarize(ann_all, "overall_allrows")

    # unique-by-SMILES
    if DO_UNIQUE_BY_SMILES:
        uniq = both.drop_duplicates(subset=[SMILES_COL]).reset_index(drop=True)
        ann_smi = annotate_dataframe(uniq, "overall_unique_smiles")
        summarize(ann_smi, "overall_unique_smiles")

    # unique-by-CAS
    if DO_UNIQUE_BY_CAS:
        cas_col = pick_cas_col(both)

        # Representative SMILES per CAS
        rep = representative_smiles_per_cas(both, cas_col=cas_col, smiles_col=SMILES_COL)
        rep = rep.rename(columns={SMILES_COL: f"{SMILES_COL}__rep"})

        # Merge representative SMILES
        merged = both.merge(rep, on=cas_col, how="left")

        # Keep rows where SMILES matches representative SMILES, or first if missing
        mask = merged[SMILES_COL] == merged[f"{SMILES_COL}__rep"]
        chosen = merged[mask].drop_duplicates(subset=[cas_col], keep="first").copy()
        if chosen.empty:
            chosen = merged.drop_duplicates(subset=[cas_col], keep="first").copy()

        # Ensure we have the correct representative SMILES in the main column
        if f"{SMILES_COL}__rep" in chosen.columns:
            chosen[SMILES_COL] = chosen[f"{SMILES_COL}__rep"].fillna(chosen[SMILES_COL])
            chosen.drop(columns=[f"{SMILES_COL}__rep"], inplace=True)

        print(f"Unique-by-CAS view: {chosen.shape} rows (unique CAS)")
        overall_unique_cas = annotate_dataframe(chosen, label="overall_unique_cas")
        _ = summarize(overall_unique_cas, label="overall_unique_cas")



if __name__ == "__main__":
    main()
