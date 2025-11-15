import pandas as pd
import numpy as np
import os
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split

# === Load dataset ===
df_all = pd.read_csv('./datasets/integrated_dataset_filled.csv')
print(f"Loaded {len(df_all)} rows.")

# === Retrieve unique SMILES ===
unique_smiles_df = df_all[['CAS', 'Canonical SMILES']].drop_duplicates().reset_index(drop=True)
print(f"Unique SMILES: {len(unique_smiles_df)}")

# === Split into SMILES without and with dots ===
df_no_dot = unique_smiles_df[~unique_smiles_df['Canonical SMILES'].str.contains('\.')].reset_index(drop=True)
df_with_dot = unique_smiles_df[unique_smiles_df['Canonical SMILES'].str.contains('\.')].reset_index(drop=True)
print(f"Without dot: {len(df_no_dot)} | With dot: {len(df_with_dot)}")

# === Initialize Mordred calculator ===
calc = Calculator(descriptors, ignore_3D=True)
desc_names = [str(d) for d in calc.descriptors]

# === Helper functions ===
def calc_mordred(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [-1]*1613
    try:
        res = calc(mol)
        return list(res.values())
    except:
        return [-1]*1613

def get_largest(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    return Chem.MolToSmiles(largest)

METAL_ATOMIC_NUMS = set([
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89,
    90, 91, 92, 93, 94
])

def extract_metals(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    metals = set()
    frags = Chem.GetMolFrags(mol, asMols=True)
    for f in frags:
        for atom in f.GetAtoms():
            if atom.GetAtomicNum() in METAL_ATOMIC_NUMS:
                metals.add(atom.GetSymbol())
    return list(metals)

# === Compute fingerprints ===

# SMILES without dots (shared across experiments)
fps_no_dot = []
for _, row in df_no_dot.iterrows():
    v = calc_mordred(row['Canonical SMILES'])
    fps_no_dot.append([row['CAS'], row['Canonical SMILES']] + v)
df_fps_no_dot = pd.DataFrame(fps_no_dot, columns=['CAS', 'Canonical SMILES'] + desc_names)

# SMILES with dots - naive
fps_with_dot_naive = []
for _, row in df_with_dot.iterrows():
    v = calc_mordred(row['Canonical SMILES'])
    fps_with_dot_naive.append([row['CAS'], row['Canonical SMILES']] + v)
df_fps_with_dot_naive = pd.DataFrame(fps_with_dot_naive, columns=['CAS', 'Canonical SMILES'] + desc_names)

# SMILES with dots - largest fragment
fps_with_dot_largest = []
metals_info = []
for _, row in df_with_dot.iterrows():
    largest = get_largest(row['Canonical SMILES'])
    metals = extract_metals(row['Canonical SMILES'])
    metals_info.append((row['CAS'], row['Canonical SMILES'], metals))
    if largest:
        v = calc_mordred(largest)
    else:
        v = [-1]*1613
    fps_with_dot_largest.append([row['CAS'], row['Canonical SMILES']] + v)
df_fps_with_dot_largest = pd.DataFrame(fps_with_dot_largest, columns=['CAS', 'Canonical SMILES'] + desc_names)

# === Merge fingerprints ===
df1 = pd.concat([df_fps_no_dot, df_fps_with_dot_naive]).reset_index(drop=True)
df2 = pd.concat([df_fps_no_dot, df_fps_with_dot_largest]).reset_index(drop=True)
df3 = df2.copy()

# === Build metal list dynamically ===
all_metals_in_data = set()
for _, _, metals in metals_info:
    all_metals_in_data.update(metals)
metal_list = sorted(all_metals_in_data)
print("Detected metals:", metal_list)

# === Add multi-hot metal encoding to df3 ===
metal_dict = {(c, s): set(m) for c, s, m in metals_info}

metal_records = []
for _, row in df3.iterrows():
    metals = metal_dict.get((row['CAS'], row['Canonical SMILES']), [])
    vector = [1 if m in metals else 0 for m in metal_list]
    metal_records.append(vector)

df_metals = pd.DataFrame(metal_records, columns=["metal_" + m for m in metal_list])
df3 = pd.concat([df3, df_metals], axis=1)

# === Clean fingerprint columns: drop >10% NaNs and fill rest ===
def clean_fp(df_fp):
    df_fp_clean = df_fp.copy()
    # Convert all descriptor columns to numeric
    for col in df_fp_clean.columns[2:]:
        df_fp_clean[col] = pd.to_numeric(df_fp_clean[col], errors='coerce')
    # Drop columns with >10% NaN
    threshold = int(len(df_fp_clean) * 0.9)
    df_fp_clean = df_fp_clean.dropna(axis=1, thresh=threshold)
    # Fill remaining NaNs with 0
    df_fp_clean[df_fp_clean.columns[2:]] = df_fp_clean[df_fp_clean.columns[2:]].fillna(0)
    return df_fp_clean

df1 = clean_fp(df1)
df2 = clean_fp(df2)
df3 = clean_fp(df3)

# === Save cleaned fingerprints ===
df1.to_csv('./datasets/mordred_dataset1.csv', index=False)
df2.to_csv('./datasets/mordred_dataset2.csv', index=False)
df3.to_csv('./datasets/mordred_dataset3.csv', index=False)

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

np.random.seed(42)


# === Split functions ===
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

    scaffold_sets = sorted(scaffold_dict.values(), key=len, reverse=True)

    test_indices = []
    train_indices = []

    test_cutoff = int(len(df) * test_size)
    current_count = 0

    for scaffold_set in scaffold_sets:
        if current_count + len(scaffold_set) <= test_cutoff:
            test_indices.extend(scaffold_set)
            current_count += len(scaffold_set)
        else:
            train_indices.extend(scaffold_set)

    return train_indices, test_indices

def greedy_group_split(df, group_col="CAS", test_size=0.2, random_state=42):
    group_sizes = df.groupby(group_col).size().reset_index(name="size")
    group_sizes = group_sizes.sample(frac=1, random_state=random_state).reset_index(drop=True)

    test_groups = []
    test_total = 0
    test_cutoff = int(len(df) * test_size)

    for _, row in group_sizes.iterrows():
        if test_total + row["size"] <= test_cutoff:
            test_groups.append(row[group_col])
            test_total += row["size"]

    test_idx = df[df[group_col].isin(test_groups)].index.tolist()
    train_idx = df[~df[group_col].isin(test_groups)].index.tolist()

    return train_idx, test_idx

# === Generate split indices once ===
train_idx_random, test_idx_random = train_test_split(df_all.index, test_size=0.2, random_state=42, shuffle=True)
train_idx_scaffold, test_idx_scaffold = scaffold_split_balanced(df_all, smiles_col="Canonical SMILES", test_size=0.2, random_state=42)
train_idx_group, test_idx_group = greedy_group_split(df_all, group_col="CAS", test_size=0.2, random_state=42)

# Save indices for reproducibility
np.save('./datasets/train_idx_random.npy', train_idx_random)
np.save('./datasets/test_idx_random.npy', test_idx_random)
np.save('./datasets/train_idx_scaffold.npy', train_idx_scaffold)
np.save('./datasets/test_idx_scaffold.npy', test_idx_scaffold)
np.save('./datasets/train_idx_group.npy', train_idx_group)
np.save('./datasets/test_idx_group.npy', test_idx_group)

# === Load fingerprint datasets ===
df_fp1 = pd.read_csv('./datasets/mordred_dataset1.csv')
df_fp2 = pd.read_csv('./datasets/mordred_dataset2.csv')
df_fp3 = pd.read_csv('./datasets/mordred_dataset3.csv')

# === Function to merge and split ===
def merge_and_save(df_fp, tag):
    df_merged = pd.merge(df_all, df_fp, on=['CAS', 'Canonical SMILES'], how='left')

    splits = {
        'random': (train_idx_random, test_idx_random),
        'scaffold': (train_idx_scaffold, test_idx_scaffold),
        'group': (train_idx_group, test_idx_group)
    }

    for split_name, (train_idx, test_idx) in splits.items():
        train_df = df_merged.loc[train_idx]
        test_df = df_merged.loc[test_idx]

        train_df.to_csv(f'./datasets/train_{tag}_{split_name}.csv', index=False)
        test_df.to_csv(f'./datasets/test_{tag}_{split_name}.csv', index=False)

        print(f"{tag}-{split_name}: Train {len(train_df)} / Test {len(test_df)}")

# === Process all three datasets ===
merge_and_save(df_fp1, 'dataset1')
merge_and_save(df_fp2, 'dataset2')
merge_and_save(df_fp3, 'dataset3')

print("All splits generated and saved.")
