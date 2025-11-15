import pandas as pd
import os
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
# Load raw datasets
df_data = pd.read_csv("./datasets/integrated_dataset_log10detect_filled.csv")
df_fp = pd.read_csv("./datasets/chemical_fingerprints_mordred_clean_0.csv")
df_fp = df_fp.drop_duplicates(subset="CAS")
if "Canonical SMILES" in df_fp.columns:
    df_fp = df_fp.drop(columns=["Canonical SMILES"])
# Merge full dataset on CAS
df_full = pd.merge(df_data, df_fp, on="CAS", how="inner")
N_total = len(df_full)

def save_split(df_train, df_test, name):
    os.makedirs("splits", exist_ok=True)
    df_train.to_csv(f"./splits/{name}_train.csv", index=False)
    df_test.to_csv(f"./splits/{name}_test.csv", index=False)



def get_bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)

def scaffold_split_balanced(df, smiles_col="Canonical SMILES", test_size=0.2, random_state=42):
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(get_bemis_murcko_scaffold)

    # Group indices by scaffold
    scaffold_dict = defaultdict(list)
    for i, row in df.iterrows():
        scaffold_dict[row["scaffold"]].append(i)

    # Randomly shuffle scaffold groups
    scaffold_sets = list(scaffold_dict.values())
    rng = np.random.RandomState(random_state)
    rng.shuffle(scaffold_sets)

    test_indices, train_indices = [], []
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


def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def similarity_split_balanced(df, smiles_col="Canonical SMILES", test_size=0.2, n_clusters=50, random_state=42):
    df = df.copy()
    
    # Generate Morgan fingerprints
    fps = [morgan_fp(s) for s in df[smiles_col]]
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    fps = [fps[i] for i in valid_indices]
    
    # Convert to numpy array
    arr = []
    for fp in fps:
        np_fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, np_fp)
        arr.append(np_fp)
    arr = np.array(arr)

    # Restrict df to valid molecules
    df = df.iloc[valid_indices].reset_index(drop=True)

    # Cluster fingerprints
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(arr)
    df["cluster"] = kmeans.labels_

    # Group by cluster
    cluster_dict = defaultdict(list)
    for i, row in df.iterrows():
        cluster_dict[row["cluster"]].append(i)

    # Sort clusters by size (optional: smallest-to-largest to balance test size)
    cluster_groups = sorted(cluster_dict.values(), key=len, reverse=True)

    np.random.seed(random_state)
    test_indices = []
    train_indices = []

    test_cutoff = int(len(df) * test_size)
    current_count = 0

    for cluster in cluster_groups:
        if current_count + len(cluster) <= test_cutoff:
            test_indices.extend(cluster)
            current_count += len(cluster)
        else:
            train_indices.extend(cluster)

    df_test = df.iloc[test_indices].drop(columns=["cluster"]).reset_index(drop=True)
    df_train = df.iloc[train_indices].drop(columns=["cluster"]).reset_index(drop=True)
    
    return df_train, df_test


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

    # Assign splits
    df_test = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    df_train = df[~df[group_col].isin(test_groups)].reset_index(drop=True)

    actual_ratio = len(df_test) / total_size
    print(f"Greedy group split: test size = {len(df_test)}, ratio = {actual_ratio:.4f}")
    
    return df_train, df_test

df_train, df_test = greedy_group_split(df_full, group_col="CAS", test_size=0.2, random_state=42)
save_split(df_train, df_test, "groupsplit")

TEST_RATIO = 0.2
print(f"GroupShuffle test ratio: {TEST_RATIO:.2f}")
df_scaffold_train, df_scaffold_test = scaffold_split_balanced(df_full, test_size=TEST_RATIO)
df_sim_train, df_sim_test = similarity_split_balanced(df_full, test_size=TEST_RATIO)
save_split(df_scaffold_train, df_scaffold_test, 'scaffold')
save_split(df_sim_train, df_sim_test, 'similarity')
print(f"Scaffold split sizes: Train = {len(df_scaffold_train)}, Test = {len(df_scaffold_test)}")
print(f"Similarity split sizes: Train = {len(df_sim_train)}, Test = {len(df_sim_test)}")


