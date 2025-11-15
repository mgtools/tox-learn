#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from gpboost import GPBoostClassifier

TOX_ORDER = [
    "Very highly toxic",
    "Highly toxic",
    "Moderately toxic",
    "Slightly toxic",
    "Practically nontoxic",
]

def classify_toxicity(log_lc50):
    lc50 = 10.0 ** log_lc50
    if lc50 < 0.1: return "Very highly toxic"
    if lc50 <= 1:  return "Highly toxic"
    if lc50 <= 10: return "Moderately toxic"
    if lc50 <= 100:return "Slightly toxic"
    return "Practically nontoxic"

def compute_max_similarity(train_smiles, test_smiles):
    def fp(smi):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
    train_fps = [fp(s) for s in train_smiles if fp(s) is not None]
    out = []
    for s in test_smiles:
        tfp = fp(s)
        if tfp is None or not train_fps:
            out.append(0.0)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
            out.append(max(sims))
    return out

def prepare_xy(df_train, df_test):
    fp_cols = df_train.columns[20:]
    taxonomy_cols = [
        "Taxonomic kingdom", "Taxonomic phylum or division", "Taxonomic subphylum",
        "Taxonomic class", "Taxonomic order", "Taxonomic family"
    ]
    numerical_cols = ["Duration (hours)"]
    feature_cols = list(numerical_cols) + list(taxonomy_cols) + list(fp_cols)

    # target
    df_train = df_train.dropna(subset=["Effect value"])
    df_test  = df_test.dropna(subset=["Effect value"])
    df_train["log_effect"] = np.log10(df_train["Effect value"])
    df_test["log_effect"]  = np.log10(df_test["Effect value"])
    df_train["tox_class"]  = df_train["log_effect"].apply(classify_toxicity)
    df_test["tox_class"]   = df_test["log_effect"].apply(classify_toxicity)

    # features
    df_train = df_train.dropna(subset=feature_cols)
    df_test  = df_test.dropna(subset=feature_cols)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), taxonomy_cols),
        ("num", SimpleImputer(strategy="constant", fill_value=0), numerical_cols),
        ("fp", "passthrough", list(fp_cols))
    ])

    return feature_cols, preprocessor, df_train, df_test

def fit_classifier(df_train, feature_cols, preprocessor):
    clf = make_pipeline(preprocessor, GPBoostClassifier())
    clf.fit(df_train[feature_cols], df_train["tox_class"])
    return clf

def get_subset(df_test, df_train, threshold, subset):
    df_test = df_test.copy()
    df_test["max_similarity"] = compute_max_similarity(df_train["Canonical SMILES"], df_test["Canonical SMILES"])
    if subset.lower() == "seen":
        return df_test[df_test["max_similarity"] >= threshold]
    else:
        return df_test[df_test["max_similarity"] < threshold]

def cm_counts_and_percentages(y_true, y_pred):
    """Return (cm_counts, cm_rowpct) with fixed label order."""
    cm = confusion_matrix(y_true, y_pred, labels=TOX_ORDER)
    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, np.maximum(row_sums, 1), where=row_sums!=0)
    return cm, pct

def annotate_matrix(cm_counts, cm_pct):
    """Build string annotations like '85% (34)'."""
    ann = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            ann[i,j] = f"{cm_pct[i,j]*100:0.0f}%\n({cm_counts[i,j]})"
    return ann

def plot_three_confusions(confusions, subset_name, out_png="combined_confusion_3splits.png"):
    """confusions: dict {SplitName: (cm_counts, cm_rowpct)}"""
    splits = ["Group", "Scaffold", "Cluster"]
    plt.figure(figsize=(18, 5))
    for k, split in enumerate(splits, start=1):
        ax = plt.subplot(1, 3, k)
        if split in confusions:
            cm_counts, cm_pct = confusions[split]
            ann = annotate_matrix(cm_counts, cm_pct)
            sns.heatmap(cm_pct, vmin=0, vmax=1, cmap="Blues",
                        xticklabels=TOX_ORDER, yticklabels=TOX_ORDER,
                        annot=ann, fmt="", cbar=(k==3), ax=ax)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{split}\n(no samples)", ha="center", va="center", fontsize=14)
            continue
        ax.set_title(f"{split} ({subset_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SAVE] {out_png}")

def build_combined_confusion(train_test_settings, threshold=0.9, subset="Unseen",
                             out_png="combined_confusion_3splits.png"):
    """
    train_test_settings: list of tuples (SplitName, train_csv, test_csv)
    subset: "Unseen" or "Seen"
    """
    confusions = {}
    for split, train_path, test_path in train_test_settings:
        print(f"\n=== {split} | subset={subset} ===")
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        feature_cols, preproc, df_train, df_test = prepare_xy(df_train, df_test)
        clf = fit_classifier(df_train, feature_cols, preproc)

        df_sub = get_subset(df_test, df_train, threshold, subset)
        n = len(df_sub)
        print(f"{split}: {subset} n={n}")
        if n == 0:
            continue

        y_true = df_sub["tox_class"]
        y_pred = clf.predict(df_sub[feature_cols])
        cm_counts, cm_pct = cm_counts_and_percentages(y_true, y_pred)
        confusions[split] = (cm_counts, cm_pct)

    if not confusions:
        print("No subsets had samples; nothing to plot.")
        return
    plot_three_confusions(confusions, subset_name=subset, out_png=out_png)

def main():
    settings = [
        ("Group",    "splits/groupsplit_train.csv",    "splits/groupsplit_test.csv"),
        ("Scaffold", "splits/scaffold_train.csv",      "splits/scaffold_test.csv"),
        ("Cluster",  "splits/similarity_train.csv",    "splits/similarity_test.csv"),
    ]

    build_combined_confusion(settings, threshold=0.9, subset="Unseen",
                             out_png="combined_confusion_unseen_3splits.png")

    # build_combined_confusion(settings, threshold=0.9, subset="Seen",
    #                          out_png="combined_confusion_seen_3splits.png")

if __name__ == "__main__":
    main()
