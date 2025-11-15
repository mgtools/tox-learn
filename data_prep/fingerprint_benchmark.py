import pandas as pd
import os
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

np.random.seed(42)
taxo_files = {
    'origin': './datasets/integrated_dataset_log10detect_filled.csv',#'./datasets/integrated_dataset_origin_bench.csv'
    # 'ncbi': './datasets/integrated_dataset_ncbi_bench.csv'
}

output_dir = './benchmarking_splits'
os.makedirs(output_dir, exist_ok=True)

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
    np.random.seed(random_state)

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

    df_train = df.loc[train_indices].drop(columns=["scaffold"])
    df_test = df.loc[test_indices].drop(columns=["scaffold"])

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def greedy_group_split(df, group_col="CAS", test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    # Group by CAS and get group sizes
    group_sizes = df.groupby(group_col).size().reset_index(name="size")
    
    # Shuffle groups
    group_sizes = group_sizes.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    total_size = len(df)
    test_cutoff = int(total_size * test_size)

    test_groups = []
    test_total = 0

    for _, row in group_sizes.iterrows():
        if test_total + row["size"] <= test_cutoff:
            test_groups.append(row[group_col])
            test_total += row["size"]

    df_test = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    df_train = df[~df[group_col].isin(test_groups)].reset_index(drop=True)

    actual_ratio = len(df_test) / total_size
    print(f"Greedy group split: test size = {len(df_test)}, ratio = {actual_ratio:.4f}")
    
    return df_train, df_test

for taxo_name, taxo_path in taxo_files.items():
    df_taxo = pd.read_csv(taxo_path, encoding='utf-8')
    print(f"Loaded {taxo_name} taxonomy: {len(df_taxo)} samples")

    # Random split
    train_rand, test_rand = train_test_split(df_taxo, test_size=0.2, random_state=42, shuffle=True)
    train_rand.to_csv(os.path.join(output_dir, f'{taxo_name}_random_train.csv'), index=False)
    test_rand.to_csv(os.path.join(output_dir, f'{taxo_name}_random_test.csv'), index=False)
    print(f"Saved random split for {taxo_name}: {len(train_rand)} train / {len(test_rand)} test")

    # Scaffold split
    train_scf, test_scf = scaffold_split_balanced(df_taxo, smiles_col="Canonical SMILES", test_size=0.2, random_state=42)
    train_scf.to_csv(os.path.join(output_dir, f'{taxo_name}_scaffold_train.csv'), index=False)
    test_scf.to_csv(os.path.join(output_dir, f'{taxo_name}_scaffold_test.csv'), index=False)
    print(f"Saved scaffold split for {taxo_name}: {len(train_scf)} train / {len(test_scf)} test")
    
    # Group split
    train_grp, test_grp = greedy_group_split(df_taxo, group_col="CAS", test_size=0.2, random_state=42)
    train_grp.to_csv(os.path.join(output_dir, f'{taxo_name}_group_train.csv'), index=False)
    test_grp.to_csv(os.path.join(output_dir, f'{taxo_name}_group_test.csv'), index=False)
    print(f"Saved group split for {taxo_name}: {len(train_grp)} train / {len(test_grp)} test")


fingerprint_files = {
    # 'morgan': './datasets/chemical_fingerprints_morgan.csv',
    # 'maccs': './datasets/chemical_fingerprints_maccs.csv',
    # 'rdkit': './datasets/chemical_fingerprints_rdkit.csv',
    'mordred': './datasets/chemical_fingerprints_mordred_clean_0.csv'
}

split_dir = './benchmarking_splits'
final_dir = './benchmarking_fingerprint'
os.makedirs(final_dir, exist_ok=True)

for taxo_name in taxo_files.keys():
    for split_type in ['random', 'scaffold', 'group']:
        train_taxo = pd.read_csv(os.path.join(split_dir, f'{taxo_name}_{split_type}_train_.csv'))
        test_taxo = pd.read_csv(os.path.join(split_dir, f'{taxo_name}_{split_type}_test.csv'))

        for fp_name, fp_path in fingerprint_files.items():
            df_fp = pd.read_csv(fp_path, encoding='utf-8')

            train_merged = pd.merge(train_taxo, df_fp, on=['CAS', 'Canonical SMILES'], how='inner')
            test_merged = pd.merge(test_taxo, df_fp, on=['CAS', 'Canonical SMILES'], how='inner')

            train_merged.to_csv(os.path.join(final_dir, f'{taxo_name}_{fp_name}_{split_type}_train.csv'), index=False)
            test_merged.to_csv(os.path.join(final_dir, f'{taxo_name}_{fp_name}_{split_type}_test.csv'), index=False)

            print(f"Saved {taxo_name}_{fp_name}_{split_type}: {len(train_merged)} train / {len(test_merged)} test")
