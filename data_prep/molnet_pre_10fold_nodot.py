#!/usr/bin/env python3
# build_molnet_flatten_nodot.py
# - Drops dotted SMILES
# - Conformer cache per unique SMILES
# - FLATTENED output: each conformer becomes its own sample (legacy 'mol' key)
# - Features: taxonomy + duration (one-hot + StandardScaler)
# - Writes group & scaffold splits for seeds 0..9

import os
import pickle
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

from data_utils import conformation_arrays

RDLogger.DisableLog('rdApp.*')

# =========================
# Config
# =========================
BASE_DIR        = r"F:/molnet_dataset_nodot_flat"   # output folder
N_CONFS_TRAIN   = 3
N_CONFS_TEST    = 1   # set to 1 if you want single conf at test
MIN_HEAVY_FOR_3D = 4
REQUIRE_CARBON   = True
SEEDS            = list(range(10))  # 0..9

# =========================
# Reproducibility (ETKDG + numpy)
# =========================
def set_repro(seed: int = 2027):
    np.random.seed(seed)

set_repro(2027)

# =========================
# IO / utils
# =========================
def _load_csv_flex(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def has_meaningful_3d(smi: str, min_heavy: int, require_c: bool) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    syms = [a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    if len(syms) < min_heavy:
        return False
    if require_c and ('C' not in syms):
        return False
    return True

def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)

def scaffold_split_balanced(df, smiles_col="Canonical SMILES", test_size=0.2, random_state=42):
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(get_bemis_murcko_scaffold)
    scaffold_dict = defaultdict(list)
    for i, row in df.iterrows():
        scaffold_dict[row["scaffold"]].append(i)

    scaffold_sets = list(scaffold_dict.values())
    rng = np.random.RandomState(random_state)
    rng.shuffle(scaffold_sets)

    test_indices, train_indices = [], []
    test_cutoff = int(len(df) * test_size)
    current = 0
    for sset in scaffold_sets:
        if len(sset) > 1:
            rng.shuffle(sset)
        if current + len(sset) <= test_cutoff:
            test_indices.extend(sset)
            current += len(sset)
        else:
            train_indices.extend(sset)

    return df.loc[train_indices].drop(columns=["scaffold"]).reset_index(drop=True), \
           df.loc[test_indices].drop(columns=["scaffold"]).reset_index(drop=True)

def greedy_group_split(df, group_col="CAS", test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    group_sizes = df.groupby(group_col).size().reset_index(name="size")
    group_sizes = group_sizes.sample(frac=1, random_state=random_state).reset_index(drop=True)
    total = len(df)
    cutoff = int(total * test_size)
    test_groups, sofar = [], 0
    for _, row in group_sizes.iterrows():
        if sofar + row["size"] <= cutoff:
            test_groups.append(row[group_col])
            sofar += row["size"]
    df_test  = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    df_train = df[~df[group_col].isin(test_groups)].reset_index(drop=True)
    return df_train, df_test


FEAT_COLS = [
    'Duration (hours)', 'Taxonomic kingdom', 'Taxonomic phylum or division',
    'Taxonomic subphylum', 'Taxonomic class', 'Taxonomic order',
    'Taxonomic family'
]

def build_features(df_feat_base: pd.DataFrame):
    """One-hot categorical + scale numeric."""
    cat_cols = [c for c in FEAT_COLS if df_feat_base[c].dtype == 'object']
    num_cols = [c for c in FEAT_COLS if c not in cat_cols]

    X = pd.get_dummies(df_feat_base[FEAT_COLS], columns=cat_cols)
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    return X.values, list(X.columns)


def _make_flat_records(df_split, conf_cache, encoder, feature_map, n_confs, suffix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records, kept, drops = [], [], dict(no_confs=0, atom_miss=0, too_big=0)

    max_atoms = int(encoder["max_atom_num"])

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Build {suffix}"):
        rid = int(row["raw_idx"])
        if rid not in conf_cache:
            drops["no_confs"] = drops.get("no_confs", 0) + 1
            continue

        confs, atom_type = conf_cache[rid]

        # Ensure atom types exist in encoder
        if not all(a in encoder["atom_type"] for a in atom_type):
            drops["atom_miss"] = drops.get("atom_miss", 0) + 1
            continue

        atom_onehot = np.array([encoder["atom_type"][a] for a in atom_type], dtype=np.float32)

        used = 0
        for j, xyz in enumerate(confs):
            if used >= n_confs:
                break
            n = len(xyz)
            if n > max_atoms:
                drops["too_big"] = drops.get("too_big", 0) + 1
                continue

            mol_arr = np.concatenate([xyz.astype(np.float32), atom_onehot], axis=1)  # [n, 3 + A]
            if n < max_atoms:
                pad = max_atoms - n
                mol_arr = np.pad(mol_arr, ((0, pad), (0, 0)), constant_values=0.0)

            records.append({
                "title": f"{row['CAS']}_{row['Latin name']}_conf{j}",
                "mol": mol_arr,
                "features": feature_map[rid],
                "Effect value": float(np.log10(float(row["Effect value"])))
            })
            used += 1

        if used > 0:
            kept.append(rid)

    # Save PKL
    pkl_path = os.path.join(out_dir, f"dataset_{suffix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save aligned CSV (one row per *molecule* kept)
    df_fp = df_split[df_split["raw_idx"].isin(kept)].drop(columns=["raw_idx"], errors="ignore")
    csv_path = os.path.join(out_dir, f"dataset_{suffix}.csv")
    df_fp.to_csv(csv_path, index=False)

    print(f"[INFO] Saved {suffix}: {len(records)} conformer-samples from {len(kept)} molecules")
    if sum(drops.values()) > 0:
        print(f"[INFO] Drops: {drops}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    df_train = _load_csv_flex('./splits/groupsplit_train.csv')
    df_test  = _load_csv_flex('./splits/groupsplit_test.csv')
    df_raw   = pd.concat([df_train, df_test], ignore_index=True)
    df_raw   = df_raw.reset_index().rename(columns={'index': 'raw_idx'})

    with open('./config/preprocess_etkdgv3.yml', 'r', encoding='utf-8') as f:
        encoder = yaml.load(f, Loader=yaml.FullLoader)['encoding']

    keep = []
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Hygiene"):
        smi = str(row['Canonical SMILES'])
        if '.' in smi:
            keep.append(False); continue
        if not has_meaningful_3d(smi, MIN_HEAVY_FOR_3D, REQUIRE_CARBON):
            keep.append(False); continue
        try:
            ev = float(row["Effect value"])
            ok_ev = (ev > 0)
        except Exception:
            ok_ev = False
        ok_basic = (not pd.isna(row.get("CAS"))) and (not pd.isna(row.get("Latin name")))
        keep.append(ok_ev and ok_basic)

    df_valid = df_raw[pd.Series(keep).values].copy().reset_index(drop=True)
    df_valid.to_csv(os.path.join(BASE_DIR, 'valid_rows_nodot.csv'), index=False)
    print(f"[INFO] After hygiene (no dots): {len(df_valid)} rows")

    X_feat, feat_names = build_features(df_valid[['raw_idx'] + FEAT_COLS].set_index('raw_idx'))
    assert X_feat.shape[0] == len(df_valid)
    feature_map = {int(rid): X_feat[i] for i, rid in enumerate(df_valid['raw_idx'])}
    print(f"[INFO] Feature dim = {X_feat.shape[1]}")

    unique_smis = sorted(set(df_valid['Canonical SMILES'].astype(str).tolist()))
    conf_cache_smiles = {}
    print("[INFO] Generating 3D conformers per unique SMILES...")
    for smi in tqdm(unique_smis, desc="Conformers(unique)"):
        try:
            good, confs, atom_type = conformation_arrays(smi, num_confs=3)
            if not good:
                continue
            if len(confs[0]) < MIN_HEAVY_FOR_3D:
                continue
            if len(confs[0]) > encoder["max_atom_num"]:
                continue
            if not all(a in encoder["atom_type"] for a in atom_type):
                continue
            conf_cache_smiles[smi] = (confs, atom_type)
        except Exception:
            continue

    conf_cache = {}
    for _, row in df_valid.iterrows():
        smi = str(row["Canonical SMILES"])
        rid = int(row["raw_idx"])
        if smi in conf_cache_smiles:
            conf_cache[rid] = conf_cache_smiles[smi]
    df_valid = df_valid[df_valid["raw_idx"].isin(conf_cache.keys())].reset_index(drop=True)
    print(f"[INFO] Rows with conformers: {len(df_valid)} (unique SMILES with 3D: {len(conf_cache_smiles)})")

    for seed in SEEDS:
        print(f"\n====== SEED {seed} (group split) ======")
        df_tr_g, df_te_g = greedy_group_split(df_valid, group_col="CAS", test_size=0.2, random_state=seed)
        _make_flat_records(df_tr_g, conf_cache, encoder, feature_map, N_CONFS_TRAIN,
                           suffix=f"group_s{seed}_train_flat", out_dir=BASE_DIR)
        _make_flat_records(df_te_g, conf_cache, encoder, feature_map, N_CONFS_TEST,
                           suffix=f"group_s{seed}_test_flat", out_dir=BASE_DIR)

        print(f"\n====== SEED {seed} (scaffold split) ======")
        df_tr_s, df_te_s = scaffold_split_balanced(df_valid, smiles_col="Canonical SMILES",
                                                   test_size=0.2, random_state=seed)
        _make_flat_records(df_tr_s, conf_cache, encoder, feature_map, N_CONFS_TRAIN,
                           suffix=f"scaffold_s{seed}_train_flat", out_dir=BASE_DIR)
        _make_flat_records(df_te_s, conf_cache, encoder, feature_map, N_CONFS_TEST,
                           suffix=f"scaffold_s{seed}_test_flat", out_dir=BASE_DIR)

    print("\n[DONE] All splits written.")

if __name__ == "__main__":
    main()
