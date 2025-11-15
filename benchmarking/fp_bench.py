#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple
from scipy.stats import chi2

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from joblib import Memory

def rmse_metric(y_true, y_pred):
    # version-agnostic RMSE
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def parse_args():
    p = argparse.ArgumentParser(description="Fast tox benchmark (Linux friendly)")
    p.add_argument("--base_dir", type=str, default="./benchmarking_fingerprint",
                   help="Folder with *_train.csv and *_test.csv files")
    p.add_argument("--results_file", type=str, default="benchmark_results_linux.csv")
    p.add_argument("--pairwise_file", type=str, default="pairwise_tests_linux.csv")
    p.add_argument("--pred_dir", type=str, default="predictions_linux")
    p.add_argument("--cache_dir", type=str, default="/tmp/sk_cache_tox")
    p.add_argument("--fast_mode", action="store_true", help="Fewer folds/search iters")
    p.add_argument("--n_iter", type=int, default=None, help="RandomizedSearchCV n_iter (override)")
    p.add_argument("--inner_folds", type=int, default=None, help="Inner CV folds (override)")
    p.add_argument("--tasks", nargs="+", default=["regression"],
                   choices=["classification", "regression"],
                   help="Which tasks to run")
    p.add_argument("--use_gpboost", default=True, action="store_true", help="Enable GPBoost models")
    p.add_argument("--bootstrap_B", type=int, default=300, help="Bootstrap samples for CIs")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


# ---------- helpers ----------
def classify_toxicity(log_lc50: float) -> str:
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


TOX_ORDER = [
    "Very highly toxic", "Highly toxic", "Moderately toxic",
    "Slightly toxic", "Practically nontoxic"
]


def within_one_bin_accuracy(y_true, y_pred, class_order=TOX_ORDER) -> float:
    label_to_index = {label: idx for idx, label in enumerate(class_order)}
    ti = np.array([label_to_index[l] for l in y_true])
    pi = np.array([label_to_index[l] for l in y_pred])
    return float((np.abs(ti - pi) <= 1).mean())


def paired_bootstrap_metric(y_true, yhat_A, yhat_B, metric_fn, B=300, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    A = np.asarray(yhat_A)
    Bhat = np.asarray(yhat_B)
    n = len(y_true)
    diffs = np.empty(B, dtype=float)
    base_diff = metric_fn(y_true, A) - metric_fn(y_true, Bhat)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = metric_fn(y_true[idx], A[idx]) - metric_fn(y_true[idx], Bhat[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(base_diff), float(lo), float(hi), float(p)


def mcnemar_test(y_true, yhat_A, yhat_B):
    y_true = np.asarray(y_true)
    A = np.asarray(yhat_A)
    B = np.asarray(yhat_B)
    correct_A = (A == y_true)
    correct_B = (B == y_true)
    b = int(np.sum(correct_A & ~correct_B))
    c = int(np.sum(~correct_A & correct_B))
    stat = (abs(b - c) - 1) ** 2 / (b + c + 1e-9)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


def onehot_sparsity_kwargs():
    """
    Handle sklearn API differences:
      - sklearn >=1.2: OneHotEncoder(sparse_output=...)
      - older sklearn: OneHotEncoder(sparse=...)
    We prefer sparse=True for speed/memory on large OHE.
    """
    try:
        _ = OneHotEncoder(sparse_output=True)
        return {"sparse_output": True}
    except TypeError:
        return {"sparse": True}


# ---------- model/space ----------
def get_models_and_spaces(task: str, fast_mode: bool, use_gpboost: bool):
    models = {}
    spaces = {}

    # base spaces (kept small for speed)
    if task == "classification":
        models = {
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        }
        spaces = {
            "Gradient Boosting": {
                "model__n_estimators": [100, 200] if fast_mode else [100, 200, 400],
                "model__learning_rate": [0.05, 0.1] if fast_mode else [0.03, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3],
                "model__subsample": [1.0, 0.8]
            },
            "Random Forest": {
                "model__n_estimators": [200] if fast_mode else [200, 400, 800],
                "model__max_depth": [None, 20] if fast_mode else [None, 12, 20, 30],
                "model__max_features": ["sqrt", "log2"],
                "model__min_samples_leaf": [1, 2],
                "model__class_weight": [None, "balanced"]
            },
        }
        if use_gpboost and HAS_GPBOOST:
            models["GPBoost"] = GPBoostClassifier()
            spaces["GPBoost"] = {
                "model__n_estimators": [200] if fast_mode else [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 4, 6],
                "model__min_data_in_leaf": [10, 20, 50],
                "model__num_leaves": [31, 63],
            }

    else:  # regression
        models = {
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        }
        spaces = {
            "Gradient Boosting": {
                "model__n_estimators": [100, 200] if fast_mode else [100, 200, 400],
                "model__learning_rate": [0.05, 0.1] if fast_mode else [0.03, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3],
                "model__subsample": [1.0, 0.8]
            },
            "Random Forest": {
                "model__n_estimators": [200] if fast_mode else [200, 400, 800],
                "model__max_depth": [None, 20] if fast_mode else [None, 12, 20, 30],
                "model__max_features": ["sqrt", "log2"],
                "model__min_samples_leaf": [1, 2],
            },
        }
        if use_gpboost and HAS_GPBOOST:
            models["GPBoost"] = GPBoostRegressor()
            spaces["GPBoost"] = {
                "model__n_estimators": [200] if fast_mode else [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 4, 6],
                "model__min_data_in_leaf": [10, 20, 50],
                "model__num_leaves": [31, 63],
            }

    return models, spaces


# optional GPBoost import (lazy to keep Linux-friendly if not installed)
HAS_GPBOOST = False
try:
    from gpboost import GPBoostRegressor, GPBoostClassifier  # type: ignore
    HAS_GPBOOST = True
except Exception:
    HAS_GPBOOST = False


# ---------- core ----------
def log_results(results_file: str, **kwargs):
    existed = os.path.exists(results_file)
    row = {k: kwargs.get(k, "") for k in [
        "dataset", "fingerprint", "split", "model", "task",
        "rmse", "r2", "accuracy", "f1", "w1b", "best_params", "cv_score"
    ]}
    pd.DataFrame([row]).to_csv(results_file, index=False, mode="a", header=not existed)


def save_predictions(pred_dir: str, key_tuple: Tuple[str, str, str, str, str],
                     ids, y_true_reg, y_true_cls, y_pred_reg, y_pred_cls):
    dataset, fingerprint, split, task, model = key_tuple
    fn = f"{dataset}__{fingerprint}__{split}__{task}__{model}.csv"
    out = pd.DataFrame({"id": ids})
    if y_true_reg is not None: out["y_true_reg"] = y_true_reg
    if y_true_cls is not None: out["y_true_cls"] = y_true_cls
    if y_pred_reg is not None: out["y_pred_reg"] = y_pred_reg
    if y_pred_cls is not None: out["y_pred_cls"] = y_pred_cls
    out.to_csv(os.path.join(pred_dir, fn), index=False)


def run_pairwise_tests(pairwise_file: str, dataset: str, fingerprint: str, split: str,
                       task: str, per_model_preds: Dict[str, Tuple[str, np.ndarray, np.ndarray]],
                       B: int, seed: int):
    existed = os.path.exists(pairwise_file)
    rows = []
    names = list(per_model_preds.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Aname, Bname = names[i], names[j]
            kind_a, y_true_a, yhat_a = per_model_preds[Aname]
            kind_b, y_true_b, yhat_b = per_model_preds[Bname]
            assert kind_a == kind_b

            if task == "classification":
                y_true = np.array(y_true_a)
                A = np.array(yhat_a)
                Bp = np.array(yhat_b)

                acc_fn = lambda yt, yp: accuracy_score(yt, yp)
                f1_fn = lambda yt, yp: f1_score(yt, yp, average='weighted')
                w1_fn = lambda yt, yp: within_one_bin_accuracy(yt, yp, TOX_ORDER)

                acc_diff, acc_lo, acc_hi, acc_p = paired_bootstrap_metric(y_true, A, Bp, acc_fn, B=B, seed=seed)
                f1_diff, f1_lo, f1_hi, f1_p = paired_bootstrap_metric(y_true, A, Bp, f1_fn, B=B, seed=seed)
                w1_diff, w1_lo, w1_hi, w1_p = paired_bootstrap_metric(y_true, A, Bp, w1_fn, B=B, seed=seed)

                stat_mc, p_mc = mcnemar_test(y_true, A, Bp)

                rows += [
                    {"dataset": dataset, "fingerprint": fingerprint, "split": split, "task": task,
                     "model_A": Aname, "model_B": Bname, "metric": "accuracy",
                     "A": acc_fn(y_true, A), "B": acc_fn(y_true, Bp),
                     "diff": acc_diff, "ci_low": acc_lo, "ci_high": acc_hi, "p_boot": acc_p,
                     "mcnemar_stat": stat_mc, "mcnemar_p": p_mc},
                    {"dataset": dataset, "fingerprint": fingerprint, "split": split, "task": task,
                     "model_A": Aname, "model_B": Bname, "metric": "f1_weighted",
                     "A": f1_fn(y_true, A), "B": f1_fn(y_true, Bp),
                     "diff": f1_diff, "ci_low": f1_lo, "ci_high": f1_hi, "p_boot": f1_p,
                     "mcnemar_stat": "", "mcnemar_p": ""},
                    {"dataset": dataset, "fingerprint": fingerprint, "split": split, "task": task,
                     "model_A": Aname, "model_B": Bname, "metric": "within_one_bin",
                     "A": w1_fn(y_true, A), "B": w1_fn(y_true, Bp),
                     "diff": w1_diff, "ci_low": w1_lo, "ci_high": w1_hi, "p_boot": w1_p,
                     "mcnemar_stat": "", "mcnemar_p": ""},
                ]
            else:
                # regression
                y_true = np.array(y_true_a)
                A = np.array(yhat_a)
                Bp = np.array(yhat_b)
                rmse_fn = lambda yt, yp: rmse_metric(yt, yp)
                r2_fn = lambda yt, yp: r2_score(yt, yp)

                rmse_diff, rmse_lo, rmse_hi, rmse_p = paired_bootstrap_metric(y_true, A, Bp, rmse_fn, B=B, seed=seed)
                r2_diff, r2_lo, r2_hi, r2_p = paired_bootstrap_metric(y_true, A, Bp, r2_fn, B=B, seed=seed)

                rows += [
                    {"dataset": dataset, "fingerprint": fingerprint, "split": split, "task": task,
                     "model_A": Aname, "model_B": Bname, "metric": "rmse",
                     "A": rmse_fn(y_true, A), "B": rmse_fn(y_true, Bp),
                     "diff": rmse_diff, "ci_low": rmse_lo, "ci_high": rmse_hi, "p_boot": rmse_p,
                     "mcnemar_stat": "", "mcnemar_p": ""},
                    {"dataset": dataset, "fingerprint": fingerprint, "split": split, "task": task,
                     "model_A": Aname, "model_B": Bname, "metric": "r2",
                     "A": r2_fn(y_true, A), "B": r2_fn(y_true, Bp),
                     "diff": r2_diff, "ci_low": r2_lo, "ci_high": r2_hi, "p_boot": r2_p,
                     "mcnemar_stat": "", "mcnemar_p": ""},
                ]

    if rows:
        pd.DataFrame(rows).to_csv(pairwise_file, index=False, mode="a", header=not existed)
        print(f"[Pairwise] wrote {len(rows)} rows → {pairwise_file}")


def run_benchmark(train_path: str, test_path: str, task: str, results_file: str, pairwise_file: str,
                  pred_dir: str, cache_dir: str, fast_mode: bool,
                  n_iter_override: int, inner_folds_override: int, use_gpboost: bool,
                  bootstrap_B: int, seed: int):

    dataset_name = os.path.basename(train_path).replace("_train.csv", "")
    parts = dataset_name.split("_")
    dataset, fingerprint, split = parts[0], parts[1], parts[2]
    print(f"\n=== {dataset} | {fingerprint} | {split} | {task} ===")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    # target
    df_train = df_train.dropna(subset=["Effect value"])
    df_test = df_test.dropna(subset=["Effect value"])

    # columns as in user’s pipeline
    taxonomy_cols = [
        "Taxonomic kingdom", "Taxonomic phylum or division",
        "Taxonomic subphylum", "Taxonomic class",
        "Taxonomic order", "Taxonomic family"
    ]
    numerical_cols = ["Duration (hours)"]

    # assume fingerprints begin at index 20
    fp_cols = df_train.columns[20:]
    feature_cols = numerical_cols + taxonomy_cols + list(fp_cols)

    # drop NA
    df_train = df_train.dropna(subset=feature_cols)
    df_test = df_test.dropna(subset=feature_cols)

    # targets
    df_train["log_effect"] = np.log10(df_train["Effect value"]).astype(np.float32)
    df_test["log_effect"] = np.log10(df_test["Effect value"]).astype(np.float32)
    y_train_reg = df_train["log_effect"].values
    y_test_reg = df_test["log_effect"].values
    y_train_cls = np.array([classify_toxicity(v) for v in y_train_reg])
    y_test_cls = np.array([classify_toxicity(v) for v in y_test_reg])

    # ids
    if "CAS Number" in df_test.columns:
        ids = df_test["CAS Number"].astype(str).values
    else:
        ids = df_test.index.values

    # OneHotEncoder arg compatibility
    ohe_kwargs = onehot_sparsity_kwargs()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs), taxonomy_cols),
        ("num", SimpleImputer(strategy="constant", fill_value=0), numerical_cols),
        ("fp", "passthrough", list(fp_cols))
    ])

    models, spaces = get_models_and_spaces(task, fast_mode=fast_mode, use_gpboost=use_gpboost)

    # inner CV/scoring
    if task == "classification":
        inner_folds = inner_folds_override if inner_folds_override else (2 if fast_mode else 3)
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        scoring = "f1_weighted"
        Xtr, ytr = df_train[feature_cols], y_train_cls
        Xte, yte = df_test[feature_cols], y_test_cls
    else:
        inner_folds = inner_folds_override if inner_folds_override else (2 if fast_mode else 3)
        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        scoring = "neg_root_mean_squared_error"
        Xtr, ytr = df_train[feature_cols], y_train_reg
        Xte, yte = df_test[feature_cols], y_test_reg

    n_iter = n_iter_override if n_iter_override else (3 if fast_mode else 12)
    per_model_preds: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}
    memory = Memory(cache_dir, verbose=0)

    for model_name, base_model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", base_model)], memory=memory)
        param_space = spaces.get(model_name, {})

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=(n_iter if param_space else 1),
            scoring=scoring,
            cv=inner_cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
            random_state=seed
        )

        search.fit(Xtr, ytr)
        best_params = search.best_params_
        cv_score = search.best_score_
        best_estimator = search.best_estimator_

        if task == "classification":
            y_pred = best_estimator.predict(Xte)
            acc = float(accuracy_score(yte, y_pred))
            f1 = float(f1_score(yte, y_pred, average='weighted'))
            w1b = within_one_bin_accuracy(yte, y_pred, TOX_ORDER)
            print(f"{model_name} | acc={acc:.3f} f1={f1:.3f} w1b={w1b:.3f} | best={json.dumps(best_params)} | cv={cv_score:.4f}")

            log_results(results_file,
                        dataset=dataset, fingerprint=fingerprint, split=split,
                        model=model_name, task=task,
                        rmse="", r2="", accuracy=acc, f1=f1, w1b=w1b,
                        best_params=json.dumps(best_params), cv_score=cv_score)

            save_predictions(pred_dir,
                             (dataset, fingerprint, split, task, model_name),
                             ids=ids,
                             y_true_reg=None, y_true_cls=yte,
                             y_pred_reg=None, y_pred_cls=y_pred)
            per_model_preds[model_name] = ("cls", yte, y_pred)

        else:
            y_pred = best_estimator.predict(Xte)
            rmse = rmse_metric(yte, y_pred)

            r2v = float(r2_score(yte, y_pred))
            print(f"{model_name} | rmse={rmse:.3f} r2={r2v:.3f} | best={json.dumps(best_params)} | cv={cv_score:.4f}")

            log_results(results_file,
                        dataset=dataset, fingerprint=fingerprint, split=split,
                        model=model_name, task=task,
                        rmse=rmse, r2=r2v, accuracy="", f1="", w1b="",
                        best_params=json.dumps(best_params), cv_score=cv_score)

            save_predictions(pred_dir,
                             (dataset, fingerprint, split, task, model_name),
                             ids=ids,
                             y_true_reg=yte, y_true_cls=None,
                             y_pred_reg=y_pred, y_pred_cls=None)
            per_model_preds[model_name] = ("reg", yte, y_pred)

    run_pairwise_tests(pairwise_file, dataset, fingerprint, split, task,
                       per_model_preds, B=bootstrap_B, seed=seed)


def run_all_benchmarks(args):
    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.pairwise_file) or ".", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # init results header if not present
    if not os.path.exists(args.results_file):
        pd.DataFrame(columns=[
            "dataset", "fingerprint", "split", "model", "task",
            "rmse", "r2", "accuracy", "f1", "w1b", "best_params", "cv_score"
        ]).to_csv(args.results_file, index=False)

    # list train files
    files = [f for f in os.listdir(args.base_dir) if f.endswith("_train.csv")]
    files.sort()
    if not files:
        print(f"[ERROR] No *_train.csv files found in {args.base_dir}")
        return

    for fname in files:
        name = fname[:-10]  # strip _train.csv
        parts = name.split("_")
        if len(parts) < 3:
            print(f"[Skip malformed] {fname}")
            continue
        train_path = os.path.join(args.base_dir, fname)
        test_path = os.path.join(args.base_dir, f"{name}_test.csv")
        if not os.path.exists(test_path):
            print(f"[Missing test] {name}")
            continue

        for task in args.tasks:
            run_benchmark(
                train_path=train_path,
                test_path=test_path,
                task=task,
                results_file=args.results_file,
                pairwise_file=args.pairwise_file,
                pred_dir=args.pred_dir,
                cache_dir=args.cache_dir,
                fast_mode=args.fast_mode,
                n_iter_override=args.n_iter if args.n_iter is not None else None,
                inner_folds_override=args.inner_folds if args.inner_folds is not None else None,
                use_gpboost=args.use_gpboost,
                bootstrap_B=args.bootstrap_B,
                seed=args.random_state
            )


if __name__ == "__main__":
    args = parse_args()
    run_all_benchmarks(args)
