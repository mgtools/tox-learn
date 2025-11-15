#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_split(df_train: pd.DataFrame, df_test: pd.DataFrame, outdir: str, name: str):
    ensure_dir(outdir)
    df_train.to_csv(os.path.join(outdir, f"{name}_train.csv"), index=False)
    df_test.to_csv(os.path.join(outdir, f"{name}_test.csv"), index=False)


def get_bemis_murcko_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=True)


def morgan_fp(smiles: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if not mol:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def fps_to_numpy(fps, n_bits=2048):
    """Convert list of ExplicitBitVect to numpy 2D array [n_samples, n_bits]."""
    arr = np.zeros((len(fps), n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def split_random(df, test_size=0.2, seed=42):
    """Plain row-wise random split (no grouping)."""
    rs = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rs.shuffle(idx)
    cutoff = int(round(test_size * len(df)))
    test_idx = idx[:cutoff]
    train_idx = idx[cutoff:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_group(df, group_col="CAS", test_size=0.2, seed=42):
    """Group-aware split by CAS using GroupShuffleSplit (random; no sorting)."""
    groups = df[group_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_scaffold(df, smiles_col="Canonical SMILES", test_size=0.2, seed=42):
    """
    Scaffold-grouped split with random group shuffle (no sorting).
    Equivalent to GroupShuffleSplit on computed Bemis-Murcko scaffold IDs.
    """
    scaffolds = df[smiles_col].apply(get_bemis_murcko_scaffold)
    # Replace missing with unique tokens to avoid cross-contamination
    scaffolds = scaffolds.fillna("NOSCAF_" + pd.Series(range(len(df))).astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=scaffolds))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_similarity(df, smiles_col="Canonical SMILES", test_size=0.2, seed=42,
                     n_bits=2048, n_clusters=50, kmeans_n_init=10):
    """
    Cluster by Morgan fingerprints with KMeans (seeded), then random group shuffle by cluster label.
    """
    # Compute Morgan fingerprints
    fps = [morgan_fp(s, radius=2, n_bits=n_bits) for s in df[smiles_col]]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_idx) < 10:
        raise ValueError("Too few valid SMILES to form similarity clusters.")
    fps_valid = [fps[i] for i in valid_idx]
    X = fps_to_numpy(fps_valid, n_bits=n_bits)

    # Bound cluster count to data size
    k = int(min(max(5, n_clusters), max(5, len(valid_idx)//50)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=kmeans_n_init)
    labels = km.fit_predict(X)

    # Prepare a groups array for all rows (invalid SMILES get unique groups so they end up in train)
    groups = np.array([None] * len(df), dtype=object)
    for i, gi in zip(valid_idx, labels):
        groups[i] = f"CL_{gi}"
    # Unique tag for invalids so they don't leak across train/test due to grouping
    for i in range(len(df)):
        if groups[i] is None:
            groups[i] = f"INV_{i}"

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
def _normalize_smiles_col(df):
    # Standardize to one column name
    candidates = ["Canonical SMILES"]
    present = [c for c in candidates if c in df.columns]
    if not present:
        return df, None
    # Pick the first found and rename to "Canonical SMILES"
    col = present[0]
    if col != "Canonical SMILES":
        df = df.rename(columns={col: "Canonical SMILES"})
    return df, "Canonical SMILES"

def merge_with_single_smiles(df_data, df_fp, on="CAS", prefer="data"):
    dd, smi_d = _normalize_smiles_col(df_data.copy())
    ff, smi_f = _normalize_smiles_col(df_fp.copy())

    # If both have SMILES, check agreement per CAS
    if smi_d and smi_f:
        # Merge temporarily only on CAS to compare SMILES
        tmp = dd[[on, smi_d]].merge(ff[[on, smi_f]], on=on, how="inner", suffixes=("_data", "_fp"))
        # Compare
        mism = (tmp[f"{smi_d}_data"].fillna("").astype(str) != tmp[f"{smi_f}_fp"].fillna("").astype(str))
        n_mism = int(mism.sum())
        n_all  = len(tmp)
        if n_mism > 0:
            print(f"[WARN] SMILES mismatch on {n_mism}/{n_all} CAS rows; keeping from '{prefer}'.")
        # Choose source
        if prefer == "data":
            keep_smiles = smi_d
            ff = ff.drop(columns=[smi_f])
        else:
            keep_smiles = smi_f
            dd = dd.drop(columns=[smi_d])
    else:
        # Only one side has SMILES; keep it
        keep_smiles = smi_d or smi_f

    # Final inner merge on CAS
    df_full = dd.merge(ff, on=on, how="inner")

    if keep_smiles is None:
        raise KeyError("No SMILES column found in either dataframe after merge.")
    if keep_smiles not in df_full.columns:
        for c in df_full.columns:
            if c.endswith("_x") or c.endswith("_y"):
                base = c[:-2]
                if base == "Canonical SMILES":
                    df_full = df_full.rename(columns={c: "Canonical SMILES"})
        if "Canonical SMILES" not in df_full.columns:
            raise KeyError("SMILES column not found after merge (unexpected).")

    return df_full, "Canonical SMILES"

# ---------------------------
# Main
# ---------------------------
def main(args):
    # Load
    df_data = pd.read_csv("./datasets/integrated_dataset_log10detect_filled.csv")
    df_fp   = pd.read_csv("./datasets/chemical_fingerprints_mordred_clean_0.csv").drop_duplicates(subset="CAS")

    # If you want to force keeping SMILES from the FP file, set prefer="fp"
    df_full, smiles_col = merge_with_single_smiles(df_data, df_fp, on="CAS", prefer="data")
    print(f"[INFO] merged rows: {len(df_full)}; using SMILES column: '{smiles_col}'")

    # Output dir
    outdir = args.outdir
    ensure_dir(outdir)

    # Seeds
    seeds = list(range(1, args.n_reps + 1))

    for rep in seeds:
        seed = args.base_seed + rep  # different seeds across reps

        # RANDOM
        tr, te = split_random(df_full, test_size=args.test_size, seed=seed)
        name = f"{args.dataset_name}_{args.fingerprint_name}_random_rep{rep:02d}"
        save_split(tr, te, outdir, name)
        print(f"[random  ] rep{rep:02d}: train={len(tr)} test={len(te)}")

        # GROUP (by CAS)
        tr, te = split_group(df_full, group_col=args.group_col, test_size=args.test_size, seed=seed)
        name = f"{args.dataset_name}_{args.fingerprint_name}_group_rep{rep:02d}"
        save_split(tr, te, outdir, name)
        print(f"[group   ] rep{rep:02d}: train={len(tr)} test={len(te)}")

        # SCAFFOLD (random group shuffle over Murcko scaffolds)
        tr, te = split_scaffold(df_full, smiles_col=args.smiles_col, test_size=args.test_size, seed=seed)
        name = f"{args.dataset_name}_{args.fingerprint_name}_scaffold_rep{rep:02d}"
        save_split(tr, te, outdir, name)
        print(f"[scaffold] rep{rep:02d}: train={len(tr)} test={len(te)}")

        # SIMILARITY (KMeans clusters over Morgan fingerprints ? group shuffle)
        tr, te = split_similarity(
            df_full,
            smiles_col=args.smiles_col,
            test_size=args.test_size,
            seed=seed,
            n_bits=args.morgan_bits,
            n_clusters=args.sim_clusters,
            kmeans_n_init=args.kmeans_n_init
        )
        name = f"{args.dataset_name}_{args.fingerprint_name}_similarity_rep{rep:02d}"
        save_split(tr, te, outdir, name)
        print(f"[similari] rep{rep:02d}: train={len(tr)} test={len(te)}")

    print(f"[DONE] Wrote splits to: {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate 10x splits with 4 methods (random, group, scaffold, similarity).")
    ap.add_argument("--data", default="./datasets/integrated_dataset_log10detect_filled.csv", help="Path to integrated_dataset_log10detect_filled.csv")
    ap.add_argument("--fp", default="./datasets/chemical_fingerprints_mordred_clean_0.csv", help="Path to chemical_fingerprints_*.csv (must include CAS; SMILES not required)")
    ap.add_argument("--dataset_name", default="toxbench", help="Name prefix for output files")
    ap.add_argument("--fingerprint_name", default="mordred", help="Name tag for output files")
    ap.add_argument("--outdir", default="./splits", help="Output folder")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_reps", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=1234)
    ap.add_argument("--group_col", default="CAS")
    ap.add_argument("--smiles_col", default="Canonical SMILES")
    # similarity split params
    ap.add_argument("--morgan_bits", type=int, default=2048)
    ap.add_argument("--sim_clusters", type=int, default=50, help="Target number of KMeans clusters (auto-bounded)")
    ap.add_argument("--kmeans_n_init", type=int, default=10)
    args = ap.parse_args()
    main(args)
