import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# Load and combine datasets
train_path = "./benchmarking_datasets/origin_mordred_random_train.csv"
test_path = "./benchmarking_datasets/origin_mordred_random_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# Filter to organic atom types only
allowed_atoms = {'C', 'O', 'N', 'H', 'P', 'S', 'F', 'Cl', 'B', 'Br', 'I'}

def is_valid_organic(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return all(atom.GetSymbol() in allowed_atoms for atom in mol.GetAtoms())
    except:
        return False

df_filtered = df_combined[df_combined["Canonical SMILES"].apply(is_valid_organic)].reset_index(drop=True)

# Scaffold split
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

# Create splits
scaffold_train, scaffold_test = scaffold_split_balanced(df_filtered)
random_split = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
random_train = random_split.iloc[:int(0.8 * len(random_split))].reset_index(drop=True)
random_test = random_split.iloc[int(0.8 * len(random_split)):].reset_index(drop=True)

# Save splits
scaffold_train.to_csv("./benchmarking_datasets/filtered_scaffold_train.csv", index=False)
scaffold_test.to_csv("./benchmarking_datasets/filtered_scaffold_test.csv", index=False)
random_train.to_csv("./benchmarking_datasets/filtered_random_train.csv", index=False)
random_test.to_csv("./benchmarking_datasets/filtered_random_test.csv", index=False)
