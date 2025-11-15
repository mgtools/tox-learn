import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def compute_fingerprints(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
    return fps

def compute_similarity_distributions(train_smiles, test_smiles, split_name):
    fps_train = compute_fingerprints(train_smiles)
    fps_test = compute_fingerprints(test_smiles)

    fps_train = [fp for fp in fps_train if fp is not None]
    fps_test = [fp for fp in fps_test if fp is not None]

    max_sims = []
    for fp_test in fps_test:
        sims = DataStructs.BulkTanimotoSimilarity(fp_test, fps_train)
        max_sims.append(np.max(sims))

    return pd.DataFrame({
        "Split": split_name,
        "Max Similarity": max_sims
    })

# Load all splits and compute similarity
splits = {
    "Random": ("./benchmarking_datasets/origin_maccs_random_train.csv", "./benchmarking_datasets/origin_maccs_random_test.csv"),
    "Group": ("splits/groupsplit_train.csv", "splits/groupsplit_test.csv"),
    "Scaffold": ("splits/scaffold_train.csv", "splits/scaffold_test.csv"),
}

frames = []
for name, (train_path, test_path) in splits.items():
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    sim_df = compute_similarity_distributions(df_train["Canonical SMILES"], df_test["Canonical SMILES"], name)
    frames.append(sim_df)

all_sims = pd.concat(frames)

# Plot violin plots
plt.figure(figsize=(6, 5))
sns.set(style="whitegrid", font="Times New Roman", font_scale=1.2)

ax = sns.violinplot(
    x="Split", y="Max Similarity", data=all_sims,
    order=["Random", "Group", "Scaffold"],
    palette="Set2", inner="box", linewidth=1
)

plt.title("Distribution of Max Tanimoto Similarity to Training Set", fontsize=13)
plt.ylabel("Max Similarity to Training Set", fontsize=12)
plt.xlabel("Split Strategy", fontsize=12)
plt.legend(loc="lower left", fontsize=10)
plt.tight_layout()

plt.savefig("max_similarity_violin.png", dpi=600)
plt.savefig("max_similarity_violin.pdf")
plt.show()
