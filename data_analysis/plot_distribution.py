#!/usr/bin/env python3
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

TOX_ORDER = [
    "Very highly toxic", "Highly toxic", "Moderately toxic",
    "Slightly toxic", "Practically nontoxic"
]

PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<split>random|group|scaffold|similarity)_rep(?P<rep>\d{2})_(?P<part>train|test)\.csv$",
    re.IGNORECASE
)

def classify_from_effect_value(df, effect_col="Effect value"):
    if effect_col not in df.columns:
        raise KeyError(f"Column '{effect_col}' not found to derive classes.")
    log_lc50 = np.log10(pd.to_numeric(df[effect_col], errors="coerce"))
    def classify(logv):
        if np.isnan(logv): return None
        lc50 = 10 ** logv
        if lc50 < 0.1:   return TOX_ORDER[0]
        elif lc50 <= 1:  return TOX_ORDER[1]
        elif lc50 <= 10: return TOX_ORDER[2]
        elif lc50 <= 100:return TOX_ORDER[3]
        else:            return TOX_ORDER[4]
    return log_lc50.apply(classify)

def proportions(series, order):
    vc = series.value_counts(normalize=True, dropna=True)
    return np.array([float(vc.get(k, 0.0)) for k in order], dtype=float)

def scan_splits(splits_dir):
    """
    Return a dict: splits[split][rep]['train'|'test'] -> filepath
    Auto-discovers any dataset/fingerprint prefix.
    """
    splits = defaultdict(lambda: defaultdict(dict))
    for fname in os.listdir(splits_dir):
        m = PATTERN.match(fname)
        if not m:
            continue
        split = m.group("split").lower()
        rep   = int(m.group("rep"))
        part  = m.group("part").lower()
        splits[split][rep][part] = os.path.join(splits_dir, fname)
    return splits

def collect_stats_for_split(split, files_for_split, class_col, effect_col):
    """
    files_for_split: dict rep -> {'train': path, 'test': path}
    Returns dict with means/stds for train/test proportions across reps.
    """
    train_props = []
    test_props  = []
    reps_used = []
    for rep, parts in sorted(files_for_split.items()):
        if ("train" not in parts) or ("test" not in parts):
            continue  
        tr = pd.read_csv(parts["train"])
        te = pd.read_csv(parts["test"])

        if class_col and (class_col in tr.columns) and (class_col in te.columns):
            tr_cls = tr[class_col].astype(str)
            te_cls = te[class_col].astype(str)
        else:
            tr_cls = classify_from_effect_value(tr, effect_col=effect_col)
            te_cls = classify_from_effect_value(te, effect_col=effect_col)

        train_props.append(proportions(tr_cls, TOX_ORDER))
        test_props.append(proportions(te_cls, TOX_ORDER))
        reps_used.append(rep)

    if not train_props or not test_props:
        return None

    train_props = np.vstack(train_props)
    test_props  = np.vstack(test_props)

    return {
        "split": split,
        "reps": reps_used,
        "train_mean": train_props.mean(axis=0),
        "train_std":  train_props.std(axis=0, ddof=1) if len(reps_used)>1 else np.zeros(5),
        "test_mean":  test_props.mean(axis=0),
        "test_std":   test_props.std(axis=0, ddof=1) if len(reps_used)>1 else np.zeros(5),
        "n_reps":     len(reps_used)
    }

def plot_overlap_hist(stats, outdir):
    split = stats["split"]
    x = np.arange(len(TOX_ORDER))
    w = 0.75  

    plt.figure(figsize=(8, 4.6))

    # Train bars
    plt.bar(x, stats["train_mean"], width=w, alpha=0.5, label=f"Train (n={stats['n_reps']})")
    plt.errorbar(x, stats["train_mean"], yerr=stats["train_std"], fmt='none', capsize=3)

    # Test bars
    plt.bar(x, stats["test_mean"], width=w, alpha=0.5, label=f"Test (n={stats['n_reps']})")
    plt.errorbar(x, stats["test_mean"], yerr=stats["test_std"], fmt='none', capsize=3)

    plt.xticks(x, TOX_ORDER, rotation=18, ha='right')
    plt.ylabel("Proportion")
    plt.ylim(0, 1.0)
    plt.title(f"Average toxicity-class distribution - {split.title()} split")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, f"avg_class_hist_{split}.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Average Train/Test class distribution per split with overlapping bars.")
    ap.add_argument("--splits_dir", default="./splits", help="Folder containing *_{split}_repXX_{train,test}.csv")
    ap.add_argument("--class_col", default=None, help="Existing class column; else derive from Effect value")
    ap.add_argument("--effect_col", default="Effect value", help="Column to derive classes from when class_col is None")
    ap.add_argument("--outdir", default="./figures_split_distributions", help="Output folder for figures")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    found = scan_splits(args.splits_dir)
    if not found:
        print(f"[ERROR] No split files found in {args.splits_dir}. "
              f"Expected pattern like '..._random_rep01_train.csv'.")
        return

    print("[INFO] Found splits & reps:")
    for split, reps in found.items():
        tr = sum(1 for r,p in reps.items() if "train" in p)
        te = sum(1 for r,p in reps.items() if "test" in p)
        print(f"  - {split}: reps with train={tr}, test={te}")

    made = []
    for split, files_for_split in found.items():
        stats = collect_stats_for_split(split, files_for_split,
                                        class_col=args.class_col,
                                        effect_col=args.effect_col)
        if stats is None:
            print(f"[WARN] Skipping split '{split}' (no complete train+test pairs).")
            continue
        out = plot_overlap_hist(stats, args.outdir)
        print(f"[OK] Wrote {out}")
        made.append(out)

    if not made:
        print("[WARN] No figures created. Check filename pattern and column names.")

if __name__ == "__main__":
    main()
