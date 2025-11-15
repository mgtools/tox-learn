#!/usr/bin/env python3 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']  
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['text.usetex'] = False              
mpl.rcParams['font.size'] = 24           # Base font size for all text
mpl.rcParams['axes.titlesize'] = 24      # Axes title font size
mpl.rcParams['axes.labelsize'] = 24      # X/Y label font size
mpl.rcParams['xtick.labelsize'] = 24     # X tick labels
mpl.rcParams['ytick.labelsize'] = 24     # Y tick labels
mpl.rcParams['legend.fontsize'] = 24     # Legend text
mpl.rcParams['figure.titlesize'] = 24 
# Reusable font handle
STAR_FP = FontProperties(family='DejaVu Sans')

# Unicode star glyphs
STAR_FILLED = u'\u2605'  
STAR_OPEN   = ''


BENCH_PATH = "benchmark_results_linux.csv"
PAIR_PATH  = "pairwise_tests_linux.csv"
OUT_DIR    = Path("plots_bar_by_split_combined")


df   = pd.read_csv(BENCH_PATH)   
pair = pd.read_csv(PAIR_PATH)  

def bootstrap_ci(group_df, value_col, n_boot=1000, alpha=0.05, random_state=123):
    """Compute mean and bootstrap CI over datasets within a group."""
    rng = np.random.default_rng(random_state)
    datasets = group_df['dataset'].unique()
    pivot = group_df.groupby('dataset')[value_col].mean().reindex(datasets)
    arr = pivot.values
    n   = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    point = float(np.nanmean(arr))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.nanmean(arr[idx]))
    boots = np.asarray(boots, dtype=float)
    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi

records = []
for (split, fingerprint, model, task), g in df.groupby(['split', 'fingerprint', 'model', 'task']):
    metric_col = 'f1' if task == 'classification' else 'rmse'
    mean, ci_lo, ci_hi = bootstrap_ci(g, metric_col)
    records.append({
        'split': split, 'fingerprint': fingerprint, 'model': model, 'task': task,
        'metric': 'F1' if task == 'classification' else 'RMSE',
        'mean': mean, 'ci_low': ci_lo, 'ci_high': ci_hi
    })
agg = pd.DataFrame.from_records(records)

dom_rows = []
for (split, fingerprint, task), g in pair.groupby(['split', 'fingerprint', 'task']):
    models = sorted(set(g['model_A']).union(set(g['model_B'])))
    winners = []
    for m in models:
        ok = True
        for other in models:
            if other == m:
                continue
            rows = g[((g['model_A'] == m) & (g['model_B'] == other)) |
                     ((g['model_A'] == other) & (g['model_B'] == m))]
            if rows.empty:
                ok = False
                break
            row = rows.iloc[0]
            p = row['p_boot']
            if task == 'classification':  # higher is better
                if row['model_A'] == m:
                    eff   = row['diff']        
                    ci_lo = row['ci_low']
                    ci_hi = row['ci_high']
                else:
                    eff   = -row['diff']       
                    ci_lo = -row['ci_high']
                    ci_hi = -row['ci_low']
                cond = (eff > 0) and (ci_lo > 0) and (p < 0.05)
            else:  
                if row['model_A'] == m:
                    eff   = -row['diff']
                    ci_lo = -row['ci_high']
                    ci_hi = -row['ci_low']
                else:
                    eff   = row['diff']
                    ci_lo = row['ci_low']
                    ci_hi = row['ci_high']
                cond = (eff > 0) and (ci_lo > 0) and (p < 0.05)
            if not cond:
                ok = False
                break
        if ok:
            winners.append(m)
    dom_rows.append({'split': split, 'fingerprint': fingerprint, 'task': task, 'dominant_models': winners})
dom = pd.DataFrame(dom_rows)

def star_type(split, fingerprint, task, model):
    rows = dom[(dom['split'] == split) & (dom['fingerprint'] == fingerprint) & (dom['task'] == task)]
    if rows.empty:
        return "open"
    winners = rows.iloc[0]['dominant_models']
    if isinstance(winners, list) and len(winners) == 1 and model in winners:
        return "filled"
    else:
        return "open"

OUT_DIR.mkdir(parents=True, exist_ok=True)

splits       = sorted(agg['split'].unique().tolist())
fingerprints = ['maccs', 'mordred', 'morgan', 'rdkit'] 
models       = sorted(agg['model'].unique().tolist())

def plot_split_combined(split):
    sub_cls = agg[(agg['split'] == split) & (agg['task'] == 'classification')].copy()
    sub_reg = agg[(agg['split'] == split) & (agg['task'] == 'regression')].copy()

    for sub in (sub_cls, sub_reg):
        sub['fingerprint'] = pd.Categorical(sub['fingerprint'], categories=fingerprints, ordered=True)
        sub.sort_values(['fingerprint', 'model'], inplace=True)

    x_positions = np.arange(len(fingerprints))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax_top, ax_bot = axes

    def panel_offsets(sub_df):
        all_means = []
        for fp in fingerprints:
            for m in models:
                r = sub_df[(sub_df['fingerprint'] == fp) & (sub_df['model'] == m)]
                if not r.empty:
                    all_means.append(r['mean'].iloc[0])
        if not all_means:
            return 0.02
        y_rng = (np.nanmax(all_means) - np.nanmin(all_means)) if np.isfinite(all_means).any() else 1.0
        return 0.02 * (y_rng if y_rng > 0 else 1.0)

    offset_cls = panel_offsets(sub_cls)
    offset_reg = panel_offsets(sub_reg)

    for j, m in enumerate(models):
        mdf = sub_cls[sub_cls['model'] == m].set_index('fingerprint').reindex(fingerprints)
        means = mdf['mean'].values
        ci_l  = mdf['ci_low'].values
        ci_h  = mdf['ci_high'].values
        offs  = x_positions - 0.4 + width/2 + j*width
        ax_top.bar(offs, means, width=width, label=m)
        yerr = np.vstack([means - ci_l, ci_h - means])
        ax_top.errorbar(offs, means, yerr=yerr, fmt='none', capsize=3, linewidth=1)

        for k, fp in enumerate(fingerprints):
            if k >= len(means) or np.isnan(means[k]):
                continue
            s_type = star_type(split, fp, 'classification', m)
            if s_type == "filled":
                ax_top.text(
                    offs[k], means[k] + offset_cls,
                    STAR_FILLED, ha='center', va='bottom', fontsize=12, fontproperties=STAR_FP
                )
            else:
                filled_exists = any(star_type(split, fp, 'classification', mm) == "filled" for mm in models)
                if not filled_exists:
                    ax_top.text(
                        offs[k], means[k] + offset_cls,
                        STAR_OPEN, ha='center', va='bottom', fontsize=12, fontproperties=STAR_FP
                    )

    ax_top.set_ylabel('F1')
    ax_top.set_title(f'Classification')
    ax_top.legend(loc='best', title='Model')

    for j, m in enumerate(models):
        mdf = sub_reg[sub_reg['model'] == m].set_index('fingerprint').reindex(fingerprints)
        means = mdf['mean'].values
        ci_l  = mdf['ci_low'].values
        ci_h  = mdf['ci_high'].values
        offs  = x_positions - 0.4 + width/2 + j*width
        ax_bot.bar(offs, means, width=width, label=m)
        yerr = np.vstack([means - ci_l, ci_h - means])
        ax_bot.errorbar(offs, means, yerr=yerr, fmt='none', capsize=3, linewidth=1)

        for k, fp in enumerate(fingerprints):
            if k >= len(means) or np.isnan(means[k]):
                continue
            s_type = star_type(split, fp, 'regression', m)
            if s_type == "filled":
                ax_bot.text(
                    offs[k], means[k] + offset_reg,
                    STAR_FILLED, ha='center', va='bottom', fontsize=12, fontproperties=STAR_FP
                )
            else:
                filled_exists = any(star_type(split, fp, 'regression', mm) == "filled" for mm in models)
                if not filled_exists:
                    ax_bot.text(
                        offs[k], means[k] + offset_reg,
                        STAR_OPEN, ha='center', va='bottom', fontsize=12, fontproperties=STAR_FP
                    )

    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels(fingerprints)
    ax_bot.set_xlabel('Fingerprint')
    ax_bot.set_ylabel('RMSE')
    ax_bot.set_title(f'Regression')

    fig.tight_layout()
    out_file = OUT_DIR / f"{split}_combined.png"
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

for sp in sorted(df['split'].unique()):
    plot_split_combined(sp)

print("Saved figures to:", sorted([str(p) for p in OUT_DIR.glob("*.png")]))
