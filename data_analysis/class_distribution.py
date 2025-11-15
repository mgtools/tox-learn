#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "./splits"  

OUTPUT_DONUT_PNG = "class_distribution_donut.png"
OUTPUT_DONUT_SVG = "class_distribution_donut.svg"
DEDUPE = True   

TOX_ORDER = [
    "Very highly toxic",
    "Highly toxic",
    "Moderately toxic",
    "Slightly toxic",
    "Practically nontoxic",
]

def classify_toxicity_from_log10(log_lc50: float) -> str:
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

def load_all_csv(base_dir: str) -> pd.DataFrame:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"BASE_DIR not found: {base_dir}")
    files = [f for f in os.listdir(base_dir) if f.endswith("_train.csv") or f.endswith("_test.csv")]
    if not files:
        raise FileNotFoundError(f"No *_train.csv or *_test.csv files found in {base_dir}")
    files.sort()
    dfs = []
    for fn in files:
        path = os.path.join(base_dir, fn)
        try:
            df = pd.read_csv(path, low_memory=False)
            df["__source_file__"] = fn
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {fn}: {e}")
    return pd.concat(dfs, axis=0, ignore_index=True)

def dedupe_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    keys = []
    if "CAS" in df.columns:
        keys.append("CAS")
    if "Latin name" in df.columns:
        keys.append("Latin name")
    if "Duration (hours)" in df.columns:
        keys.append("Duration (hours)")
    if DEDUPE and keys:
        before = len(df)
        df = df.drop_duplicates(subset=keys)
        after = len(df)
        print(f"[INFO] Deduped by {keys}: {before} -> {after}")
    return df

def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Effect value"]).copy()
    df["log_effect"] = np.log10(df["Effect value"].astype(float))
    df["tox_class"] = df["log_effect"].apply(classify_toxicity_from_log10)
    df["tox_class"] = pd.Categorical(df["tox_class"], categories=TOX_ORDER, ordered=True)
    return df

def make_donut(counts: pd.Series, out_png: str, out_svg: str):
    total = int(counts.sum())
    sizes = counts.values.tolist()
    labels = counts.index.tolist()
    perc   = (counts / max(total, 1) * 100.0).round(1).tolist()

    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=300)

    def autopct_format(pct):
        return f"{pct:.1f}%"

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,                
        autopct=autopct_format,
        startangle=90,
        counterclock=False,
        pctdistance=0.78,
        wedgeprops=dict(linewidth=0.75, edgecolor="white"),
        textprops=dict(fontsize=10)
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    ax.axis("equal")

    kw = dict(arrowprops=dict(arrowstyle="-", lw=0.6), va="center", ha="center")
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2.0
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        label_text = f"{labels[i]}\n{sizes[i]:,} ({perc[i]}%)"
        ax.annotate(label_text, xy=(x*0.95, y*0.95), xytext=(x*1.28, y*1.28), fontsize=9, **kw)

    ax.set_title("Overall Distribution of LC50-Based Toxicity Classes", fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_svg}")
    print(f"[OK] Saved: {out_png}")
    print(f"[INFO] Total N = {total}")
    for name, c, p in zip(labels, sizes, perc):
        print(f"  {name:>22s}: {c:>6d}  ({p:>4.1f}%)")

if __name__ == "__main__":
    df_all = load_all_csv(BASE_DIR)
    df_all = dedupe_if_possible(df_all)
    df_lab = prepare_labels(df_all)
    counts = df_lab["tox_class"].value_counts().reindex(TOX_ORDER).fillna(0).astype(int)
    make_donut(counts, OUTPUT_DONUT_PNG, OUTPUT_DONUT_SVG)
