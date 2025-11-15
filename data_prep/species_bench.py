
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)

def scaffold_split(df, smiles_col="Canonical SMILES", test_size=0.2, seed=42):
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(get_scaffold)
    scaffold_dict = defaultdict(list)
    for i, row in df.iterrows():
        scaffold_dict[row["scaffold"]].append(i)
    scaffold_sets = sorted(scaffold_dict.values(), key=len, reverse=True)
    np.random.seed(seed)
    train_idx, test_idx = [], []
    cutoff = int(len(df) * test_size)
    count = 0
    for s in scaffold_sets:
        if count + len(s) <= cutoff:
            test_idx += s
            count += len(s)
        else:
            train_idx += s
    return df.loc[train_idx].drop(columns="scaffold"), df.loc[test_idx].drop(columns="scaffold")

def group_split(df, group_col, test_size=0.2, seed=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx], df.iloc[test_idx]

#  Load Data 
data_dir = './bench_species_matched'
datasets = {
    'origin': pd.read_csv(os.path.join(data_dir, 'origin_mordred_benchmark.csv')),
    'ncbi': pd.read_csv(os.path.join(data_dir, 'ncbi_mordred_benchmark.csv')),
    'deb': pd.read_csv(os.path.join(data_dir, 'deb_mordred_benchmark.csv')),
    'ablation': pd.read_csv(os.path.join(data_dir, 'ablation_mordred_benchmark.csv')),
}

#Generate unique keys to match samples
for name, df in datasets.items():
    df['key'] = df['CAS'].astype(str) + '_' + df['Latin name'].astype(str) + '_' + df['Duration (hours)'].astype(str)
    datasets[name] = df

# Find common keys across all datasets
common_keys = set.intersection(*(set(df['key']) for df in datasets.values()))
print(f"Common samples across all datasets: {len(common_keys)}")

# Filter all datasets to shared keys 
for name in datasets:
    datasets[name] = datasets[name][datasets[name]['key'].isin(common_keys)].reset_index(drop=True)

# se origin as reference for splitting 
df_origin = datasets['origin']
df_origin_random_train, df_origin_random_test = train_test_split(df_origin, test_size=0.2, random_state=42)
df_origin_scf_train, df_origin_scf_test = scaffold_split(df_origin)
df_origin_grp_train, df_origin_grp_test = group_split(df_origin, group_col="Latin name")

# Get split keys
split_keys = {
    'random_train': set(df_origin_random_train['key']),
    'random_test': set(df_origin_random_test['key']),
    'scaffold_train': set(df_origin_scf_train['key']),
    'scaffold_test': set(df_origin_scf_test['key']),
    'group_train': set(df_origin_grp_train['key']),
    'group_test': set(df_origin_grp_test['key']),
}

# Save consistent splits for all datasets 
output_dir = './benchmarking_splits_matched'
os.makedirs(output_dir, exist_ok=True)

for name, df in datasets.items():
    for split in ['random', 'scaffold', 'group']:
        df_train = df[df['key'].isin(split_keys[f'{split}_train'])].drop(columns='key')
        df_test = df[df['key'].isin(split_keys[f'{split}_test'])].drop(columns='key')
        df_train.to_csv(os.path.join(output_dir, f'{name}_{split}_train.csv'), index=False)
        df_test.to_csv(os.path.join(output_dir, f'{name}_{split}_test.csv'), index=False)
        print(f"Saved {name} {split} split: {len(df_train)} train / {len(df_test)} test")
