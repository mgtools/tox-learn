import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

#  Paths 
split_dir = './benchmarking_splits_matched'
RESULTS_FILE = './species_representation_benchmark_results_2.csv'

#  Classify Toxicity 
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

def within_one_bin_accuracy(y_true, y_pred, class_order=None):
    if class_order is None:
        class_order = ["Very highly toxic", "Highly toxic", "Moderately toxic", "Slightly toxic", "Practically nontoxic"]
    label_to_index = {label: idx for idx, label in enumerate(class_order)}
    true_idx = [label_to_index[y] for y in y_true]
    pred_idx = [label_to_index[y] for y in y_pred]
    correct = sum(abs(t - p) <= 1 for t, p in zip(true_idx, pred_idx))
    return correct / len(y_true)

def log_results(dataset, split, task, rmse, r2, acc, f1, within1bin=""):
    new_row = pd.DataFrame([{
        "dataset": dataset,
        "split": split,
        "task": task,
        "rmse": rmse,
        "r2": r2,
        "accuracy": acc,
        "f1": f1,
        "within1bin": within1bin
    }])
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(RESULTS_FILE, index=False)

#  Run Benchmark 
for file in os.listdir(split_dir):
    if file.endswith('_train.csv'):
        dataset_name = file.replace('_random_train.csv', '').replace('_scaffold_train.csv', '').replace('_group_train.csv', '')
        if 'random' in file:
            split = 'random'
        elif 'scaffold' in file:
            split = 'scaffold'
        elif 'group' in file:
            split = 'group'
        else:
            split = 'unknown'

        train_path = os.path.join(split_dir, file)
        test_path = os.path.join(split_dir, file.replace('_train.csv', '_test.csv'))

        print(f"\n Running benchmark: {dataset_name} | {split} ")

        # Load data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Drop missing targets
        df_train = df_train.dropna(subset=["Effect value"])
        df_test = df_test.dropna(subset=["Effect value"])


        #  Feature Columns by Dataset 
        if dataset_name == 'origin':
            cat_cols = ['Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
                        'Taxonomic class', 'Taxonomic order', 'Taxonomic family']
            num_cols = ['Duration (hours)']
            fp_cols = df_train.columns[20:]
            feature_cols = num_cols + cat_cols + list(fp_cols)

        elif dataset_name == 'deb':
            cat_cols = ['family', 'order', 'class', 'phylum']
            num_cols = ['Duration (hours)']
            deb_num_cols = df_train.columns[20:36] 
            fp_cols = df_train.columns[36:]
            feature_cols = num_cols + list(cat_cols) + list(deb_num_cols) + list(fp_cols)

        elif dataset_name == 'ablation':
            cat_cols = [col for col in df_train.columns if col.startswith("species_")]
            num_cols = ['Duration (hours)']
            fp_start = df_train.columns.get_loc(cat_cols[-1]) + 1
            fp_cols = df_train.columns[fp_start:]
            feature_cols = num_cols + cat_cols + list(fp_cols)


        elif dataset_name == 'ncbi':
            cat_cols = ['Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
                        'Taxonomic class', 'Taxonomic order', 'Taxonomic family']
            num_cols = ['Duration (hours)']
            fp_cols = df_train.columns[20:]
            feature_cols = num_cols + cat_cols + list(fp_cols)

        else:
            print(f"Unknown dataset type: {dataset_name}")
            continue

        # Targets
        df_train["log_effect"] = np.log10(df_train["Effect value"])
        df_test["log_effect"] = np.log10(df_test["Effect value"])
        y_train_reg = df_train["log_effect"]
        y_test_reg = df_test["log_effect"]
        y_train_cls = [classify_toxicity(val) for val in y_train_reg]
        y_test_cls = [classify_toxicity(val) for val in y_test_reg]
        # Drop missing feature rows
        df_train = df_train.dropna(subset=feature_cols)
        df_test = df_test.dropna(subset=feature_cols)
        print(df_train.shape)
        print(df_test.shape)
        # Preprocessor
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), [col for col in cat_cols if col in df_train.columns]),
            ("num", SimpleImputer(strategy="constant", fill_value=0), [col for col in num_cols if col in df_train.columns]),
            ("fp", "passthrough", [col for col in fp_cols if col in df_train.columns])
        ])

        # #  Regression 
        # reg_model = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))
        # reg_model.fit(df_train[feature_cols], y_train_reg)
        # y_pred_reg = reg_model.predict(df_test[feature_cols])
        # rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        # r2 = r2_score(y_test_reg, y_pred_reg)
        # print(f"Random Forest Regression | RMSE: {rmse:.3f} | R2: {r2:.3f}")
        # log_results(dataset_name, split, 'regression', rmse, r2, '', '')

        #  Classification 
        cls_model = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))
        cls_model.fit(df_train[feature_cols], y_train_cls)
        y_pred_cls = cls_model.predict(df_test[feature_cols])
        acc = accuracy_score(y_test_cls, y_pred_cls)
        f1 = f1_score(y_test_cls, y_pred_cls, average='weighted')
        w1b_acc = within_one_bin_accuracy(y_test_cls, y_pred_cls)
        print(f"Random Forest Classification | Accuracy: {acc:.3f} | F1: {f1:.3f} | Within-1-Bin Acc: {w1b_acc:.3f}")
        log_results(dataset_name, split, 'classification', '', '', acc, f1, within1bin=w1b_acc)
