#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from gpboost import GPBoostRegressor, GPBoostClassifier


SPLIT_TOKENS = ("random", "group", "scaffold", "similarity")
FP_TOKENS    = ("mordred", "morgan", "maccs", "rdkit")

# Multiple patterns to handle different naming schemes
PATTERNS = [
    re.compile(r"""
        ^(?P<dataset>.+?)_
        (?P<fingerprint>mordred|morgan|maccs|rdkit)_
        (?P<split>random|group|scaffold|similarity)
        (?:_(?:rep(?P<rep>\d{2})|s(?P<seed>\d+)))?
        .*_train.*\.csv$
    """, re.IGNORECASE | re.VERBOSE),

    re.compile(r"""
        ^(?P<dataset>.+?)_
        (?P<split>random|group|scaffold|similarity)
        (?:_(?:rep(?P<rep>\d{2})|s(?P<seed>\d+)))?
        .*_train.*\.csv$
    """, re.IGNORECASE | re.VERBOSE),

    re.compile(r"""
        ^(?P<prefix>.+)
        (?P<split>random|group|scaffold|similarity)
        .*_train.*\.csv$
    """, re.IGNORECASE | re.VERBOSE),
]

def parse_context_from_fname(train_fname: str):
    """Parse dataset/fingerprint/split/rep/component from a train filename."""
    base = os.path.basename(train_fname)
    dataset    = "unknown"
    fingerprint= "na"  
    split      = "unknown"
    rep        = "na"
    component  = "na"

    m = None
    for pat in PATTERNS:
        m = pat.match(base)
        if m:
            break

    if m:
        gd = m.groupdict()

        # dataset
        if gd.get("dataset"):
            dataset = gd["dataset"]
        elif gd.get("prefix"):
            dataset = gd["prefix"].rstrip("_")

        # split
        if gd.get("split"):
            split = gd["split"].lower()

        # fingerprint
        if gd.get("fingerprint"):
            fingerprint = gd["fingerprint"].lower()

        if gd.get("rep"):
            rep = f"rep{gd['rep']}"
        elif gd.get("seed"):
            rep = f"s{gd['seed']}"

        low = base.lower()
        if "largest" in low:
            component = "largest"
        elif "original" in low:
            component = "original"
    else:
        low = base.lower()
        split = next((t for t in SPLIT_TOKENS if t in low), "unknown")
        fingerprint = next((t for t in FP_TOKENS if t in low), "na")
        dataset = base.split("_")[0] or "unknown"
        mr = re.search(r"(?:_rep(\d{2})|_s(\d+))", low)
        if mr:
            rep = "rep"+(mr.group(1) or "") if mr.group(1) else "s"+mr.group(2)

    return dict(dataset=dataset, fingerprint=fingerprint, split=split, rep=rep, component=component)


def guess_fingerprint_from_columns(df: pd.DataFrame) -> str:
    """Best-effort fingerprint guess if not present in filename."""
    cols = set(df.columns.str.lower())

    mordred_hints = {"mdec-23", "ats0m", "mwc09"}
    maccs_hint = any(c.startswith("maccs_") for c in cols)
    rdkit_hint = any(c.startswith("rdkitfp_") for c in cols)
    morgan_hint = any(c.startswith("morgan_") or c.startswith("ecfp") for c in cols)
    if any(h in cols for h in mordred_hints):
        return "mordred"
    if maccs_hint:  return "maccs"
    if rdkit_hint:  return "rdkit"
    if morgan_hint: return "morgan"
    num_cols = df.columns[20:]
    if len(num_cols) > 256:
        return "morgan"
    return "na"


def _read_csv_flex(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def rmse_metric(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def classify_toxicity(log10_lc50: float) -> int:
    """Binning used elsewhere in your code: log10(LC50) -> 0..4"""
    if log10_lc50 < -1:  return 0
    elif log10_lc50 < 0: return 1
    elif log10_lc50 < 1: return 2
    elif log10_lc50 < 2: return 3
    else:                return 4


def within_one_bin_correct(y_true_bin, y_pred_bin):
    return (np.abs(np.array(y_pred_bin) - np.array(y_true_bin)) <= 1).astype(int)


def _ohe_kwargs():
    """Compat with sklearn versions (sparse_output vs sparse)."""
    try:
        _ = OneHotEncoder(sparse_output=True)
        return dict(sparse_output=True, handle_unknown="ignore")
    except TypeError:
        return dict(sparse=True, handle_unknown="ignore")


def _align_feature_columns(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols):
    """Ensure both sides have all features (missing numeric/fp cols = 0)."""
    for c in feature_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0
        if c not in df_test.columns:
            df_test[c] = 0.0
    return df_train, df_test


def _safe_info_cols(df: pd.DataFrame):
    keep = []
    for c in ["CAS", "Latin name", "Duration (hours)", "Canonical SMILES"]:
        if c in df.columns:
            keep.append(c)
    return keep


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_benchmark(train_path, test_path, task='regression', out_dir="./analysis_outputs", dedup=False):
    ctx = parse_context_from_fname(train_path)
    dataset     = ctx["dataset"]
    fingerprint = ctx["fingerprint"]
    split       = ctx["split"]
    rep         = ctx["rep"]
    component   = ctx["component"]

    print(f"\n=== {dataset}__{fingerprint}__{split}__{rep}__{component} | task={task} ===")

    df_train = _read_csv_flex(train_path)
    df_test  = _read_csv_flex(test_path)

    if fingerprint == "na":
        fingerprint = guess_fingerprint_from_columns(df_train)

    if "Effect value" not in df_train.columns or "Effect value" not in df_test.columns:
        raise KeyError("Column 'Effect value' is required in both train/test.")

    df_train = df_train.dropna(subset=["Effect value"]).reset_index(drop=True)
    df_test  = df_test.dropna(subset=["Effect value"]).reset_index(drop=True)

    if dedup:
        id_cols = [c for c in ["CAS", "Latin name", "Duration (hours)"] if c in df_train.columns]
        if id_cols:
            df_train = df_train.drop_duplicates(subset=id_cols).reset_index(drop=True)
            df_test  = df_test.drop_duplicates(subset=id_cols).reset_index(drop=True)

    taxonomy_cols = [
        "Taxonomic kingdom", "Taxonomic phylum or division", "Taxonomic subphylum",
        "Taxonomic class", "Taxonomic order", "Taxonomic family"
    ]
    numerical_cols = ["Duration (hours)"]

    fp_cols = [c for c in df_train.columns[20:] if c in df_test.columns]
    feature_cols = [c for c in (numerical_cols + taxonomy_cols + list(fp_cols)) if c in df_train.columns]

    df_train = df_train.dropna(subset=feature_cols)
    df_test  = df_test.dropna(subset=feature_cols)
    df_train, df_test = _align_feature_columns(df_train, df_test, feature_cols)

    df_train["log_effect"] = np.log10(pd.to_numeric(df_train["Effect value"], errors="coerce"))
    df_test["log_effect"]  = np.log10(pd.to_numeric(df_test["Effect value"],  errors="coerce"))
    y_train_reg = df_train["log_effect"].values
    y_test_reg  = df_test["log_effect"].values

    y_train_cls = np.array([classify_toxicity(v) for v in y_train_reg], dtype=int)
    y_test_cls  = np.array([classify_toxicity(v) for v in y_test_reg], dtype=int)

    print("Class balance (train):", np.bincount(y_train_cls, minlength=5))
    print("Class balance (test) :", np.bincount(y_test_cls,  minlength=5))

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(**_ohe_kwargs()), taxonomy_cols),
        ("num", SimpleImputer(strategy="constant", fill_value=0), numerical_cols),
        ("fp",  "passthrough", list(fp_cols))
    ])

    info_cols = _safe_info_cols(df_test)

    split_dir = os.path.join(out_dir, split if split != "unknown" else "misc")
    _ensure_dir(split_dir)

    if task == 'regression':
        model = GPBoostRegressor(random_state=42)
        pipe = make_pipeline(preprocessor, model)
        pipe.fit(df_train[feature_cols], y_train_reg)
        y_pred_reg = pipe.predict(df_test[feature_cols])

        rmse = rmse_metric(y_test_reg, y_pred_reg)
        r2   = float(r2_score(y_test_reg, y_pred_reg))

        y_pred_bin    = np.array([classify_toxicity(v) for v in y_pred_reg], dtype=int)
        exact_correct = (y_pred_bin == y_test_cls).astype(int)
        w1b_correct   = within_one_bin_correct(y_test_cls, y_pred_bin)

        exact_acc = float(exact_correct.mean())
        w1b_acc   = float(w1b_correct.mean())

        print(f"[regression] RMSE={rmse:.3f} R2={r2:.3f} | Acc={exact_acc:.3f} W1B={w1b_acc:.3f}")

        pred_df = df_test[info_cols].copy()
        pred_df["y_true_log10lc50"] = y_test_reg
        pred_df["y_pred_log10lc50"] = y_pred_reg
        pred_df["y_true_bin"] = y_test_cls
        pred_df["y_pred_bin"] = y_pred_bin
        pred_df["correct_exact"] = exact_correct
        pred_df["correct_w1b"]   = w1b_correct
  
        return {
            "task":"regression",
            "rmse":rmse,
            "r2":r2,
            "accuracy": exact_acc,
            "w1b_accuracy": w1b_acc,
            "split":split,"rep":rep,"component":component
        }

    elif task == 'classification':
        model = GPBoostClassifier(objective="multiclass", num_class=5, random_state=42)
        pipe = make_pipeline(preprocessor, model)
        pipe.fit(df_train[feature_cols], y_train_cls)
        y_pred_cls = pipe.predict(df_test[feature_cols]).astype(int)

        acc = float(accuracy_score(y_test_cls, y_pred_cls))
        f1  = float(f1_score(y_test_cls, y_pred_cls, average='weighted'))

        w1b_correct = within_one_bin_correct(y_test_cls, y_pred_cls)
        w1b_acc     = float(w1b_correct.mean())

        print(f"[classification] ACC={acc:.3f} F1={f1:.3f} W1B={w1b_acc:.3f}")

        pred_df = df_test[info_cols].copy()
        pred_df["y_true_bin"]   = y_test_cls
        pred_df["y_pred_bin"]   = y_pred_cls
        pred_df["correct_exact"]= (y_pred_cls == y_test_cls).astype(int)
        pred_df["correct_w1b"]  = w1b_correct

        return {
            "task":"classification",
            "accuracy":acc,
            "f1":f1,
            "w1b_accuracy": w1b_acc,
            "split":split,"rep":rep,"component":component
        }

    else:
        raise ValueError(f"Unknown task: {task}")
def w1b_accuracy(y_true_bin, y_pred_bin) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred_bin) - np.asarray(y_true_bin)) <= 1))


def run_all_organic_benchmarks(base_dir, out_dir="./analysis_outputs",
                               tasks=('regression','classification')):
    _ensure_dir(out_dir)
    files = sorted([f for f in os.listdir(base_dir)
                    if f.lower().endswith(".csv") and "_train" in f.lower()])
    all_metrics = []

    for train_fname in files:
        test_fname = re.sub(r"_train", "_test", train_fname, flags=re.IGNORECASE)
        train_path = os.path.join(base_dir, train_fname)
        test_path  = os.path.join(base_dir, test_fname)

        if not os.path.exists(test_path):
            print(f"[SKIP] test file missing for {train_fname}")
            continue

        for task in tasks:
            m = run_benchmark(train_path, test_path, task=task, out_dir=out_dir)
            if m is not None:
                low = train_fname.lower()
                mr = re.search(r"(?:_rep(\d{2})|_s(\d+))", low)
                seed = f"rep{mr.group(1)}" if (mr and mr.group(1)) else (f"s{mr.group(2)}" if (mr and mr.group(2)) else "na")
                m["seed"] = seed
                m["file"] = train_fname
                all_metrics.append(m)

    if all_metrics:
        runs_df = pd.DataFrame(all_metrics)
        runs_csv = os.path.join(out_dir, "gpboost_benchmark_runs.csv")
        runs_df.to_csv(runs_csv, index=False)
        print(f"[SAVE] {runs_csv}")


def main():
    ap = argparse.ArgumentParser(description="Run GPBoost regression/classification on train/test CSV pairs and save full predictions.")
    ap.add_argument("--base_dir", default="F:/molnet_dataset_nodot_flat", help="Folder containing *_train*.csv and matching *_test*.csv files")
    ap.add_argument("--out_dir", default="./analysis_outputs", help="Output folder for predictions and metrics")
    ap.add_argument("--tasks", nargs="+", default=["regression","classification"], choices=["regression","classification"])
    ap.add_argument("--dedup", action="store_true", help="Optional: drop duplicates by (CAS, Latin name, Duration)")
    args = ap.parse_args()

    run_all_organic_benchmarks(args.base_dir, out_dir=args.out_dir, tasks=tuple(args.tasks))


if __name__ == "__main__":
    main()
