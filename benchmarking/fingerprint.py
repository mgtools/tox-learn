import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from gpboost import GPModel, GPBoostRegressor, GPBoostClassifier
from tabpfn import TabPFNClassifier
import torch
from sklearn.decomposition import PCA

RESULTS_FILE = "benchmark_results_3.csv"

# Initialize results file
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("dataset,fingerprint,split,model,task,rmse,r2,accuracy,f1\n")

def classify_toxicity(log_lc50):
    lc50 = 10 ** log_lc50
    if lc50 < 0.1:
        return "Very highly toxic"
    elif lc50 <= 1:
        return "Highly toxic"
    elif lc50 <= 10:
        return "Moderately toxic"
    elif lc50 <= 100:
        return "Slightly toxic"
    else:
        return "Practically nontoxic"
# Define toxicity order globally
tox_order = ["Very highly toxic", "Highly toxic", "Moderately toxic", "Slightly toxic", "Practically nontoxic"]

def within_one_bin_accuracy(y_true, y_pred, class_order):
    label_to_index = {label: idx for idx, label in enumerate(class_order)}
    true_indices = [label_to_index[label] for label in y_true]
    pred_indices = [label_to_index[label] for label in y_pred]
    correct = sum(abs(t - p) <= 1 for t, p in zip(true_indices, pred_indices))
    return correct / len(y_true)
def log_results(**kwargs):
    with open(RESULTS_FILE, "a") as f:
        f.write(",".join(str(kwargs.get(k, "")) for k in ["dataset", "fingerprint", "split", "model", "task", "rmse", "r2", "accuracy", "f1"]) + "\n")

def run_benchmark(train_path, test_path, task='regression'):
    dataset_name = os.path.basename(train_path).replace("_train.csv", "")
    parts = dataset_name.split("_")
    dataset = parts[0]
    fingerprint = parts[1]
    split = parts[2]

    print(f"\n=== Running benchmark on: {dataset} | {fingerprint} | {split} | {task} ===")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Drop rows with missing target
    df_train = df_train.dropna(subset=["Effect value"])
    df_test = df_test.dropna(subset=["Effect value"])

    # Columns
    taxonomy_cols = [
        "Taxonomic kingdom", "Taxonomic phylum or division",
        "Taxonomic subphylum", "Taxonomic class", "Taxonomic order", "Taxonomic family"
    ]
    numerical_cols = ["Duration (hours)"]
    fp_cols = df_train.columns[20:]
    feature_cols = numerical_cols + taxonomy_cols + list(fp_cols)

    # Drop rows with missing features
    df_train = df_train.dropna(subset=feature_cols)
    df_test = df_test.dropna(subset=feature_cols)

    # Targets
    df_train["log_effect"] = np.log10(df_train["Effect value"])
    df_test["log_effect"] = np.log10(df_test["Effect value"])
    y_train_reg = df_train["log_effect"].values
    y_test_reg = df_test["log_effect"].values
    y_train_cls = [classify_toxicity(val) for val in y_train_reg]
    y_test_cls = [classify_toxicity(val) for val in y_test_reg]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), taxonomy_cols),
        ("num", SimpleImputer(strategy="constant", fill_value=0), numerical_cols),
        ("fp", "passthrough", list(fp_cols))
    ])

    # Models
    models = {
        'Gradient Boosting': (GradientBoostingRegressor(random_state=42), GradientBoostingClassifier(random_state=42)),
        'Random Forest': (RandomForestRegressor(random_state=42), RandomForestClassifier(random_state=42)),
        'GPBoost': (GPBoostRegressor(), GPBoostClassifier())
    }

    for model_name, (reg_model, cls_model) in models.items():
        if task == 'regression':
            pipe = make_pipeline(preprocessor, reg_model)
            pipe.fit(df_train[feature_cols], y_train_reg)
            y_pred = pipe.predict(df_test[feature_cols])
            rmse = mean_squared_error(y_test_reg, y_pred, squared=False)
            r2 = r2_score(y_test_reg, y_pred)
            print(f"{model_name} - RMSE: {rmse:.3f} | R2: {r2:.3f}")
            log_results(dataset=dataset, fingerprint=fingerprint, split=split, model=model_name, task="regression", rmse=rmse, r2=r2, accuracy="", f1="")

        elif task == 'classification':
            pipe = make_pipeline(preprocessor, cls_model)
            pipe.fit(df_train[feature_cols], y_train_cls)
            y_pred = pipe.predict(df_test[feature_cols])
            acc = accuracy_score(y_test_cls, y_pred)
            f1 = f1_score(y_test_cls, y_pred, average='weighted')
            wob_acc = within_one_bin_accuracy(y_test_cls, y_pred, tox_order)
            print(f"{model_name} - Accuracy: {acc:.3f} | F1: {f1:.3f} | Within-1-Bin Acc: {wob_acc:.3f}")
            log_results(dataset=dataset, fingerprint=fingerprint, split=split, model=model_name, task="classification", rmse="", r2="", accuracy=acc, f1=f1)
def run_all_benchmarks(base_dir='./benchmarking_fingerprint'):
    if os.path.exists(RESULTS_FILE):
        completed = pd.read_csv(RESULTS_FILE)
    else:
        completed = pd.DataFrame(columns=["dataset", "fingerprint", "split", "model", "task", "rmse", "r2", "accuracy", "f1"])

    files = [f for f in os.listdir(base_dir) if f.endswith('_train.csv')]

    for fname in files:
        name = fname.replace('_train.csv', '')
        parts = name.split("_")
        dataset = parts[0]
        fingerprint = parts[1]
        split = parts[2]
        train_path = os.path.join(base_dir, fname)
        test_path = os.path.join(base_dir, f"{name}_test.csv")

        for task in ['classification']: # 'regression'
            # Check if all 3 models for this dataset+task+split are done
            task_done = completed[
                (completed['dataset'] == dataset) &
                (completed['fingerprint'] == fingerprint) &
                (completed['split'] == split) &
                (completed['task'] == task)
            ]
            if len(task_done) >= 3:
                print(f"Skipping {dataset} | {fingerprint} | {split} | {task} (all models logged)")
                continue

            run_benchmark(train_path, test_path, task=task)


if __name__ == "__main__":
    run_all_benchmarks()
