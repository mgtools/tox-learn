#!/usr/bin/env python3
# plot_gpboost_mordred_by_split.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

BENCH_PATH = "benchmark_results_linux.csv"
OUT_DIR    = Path("plots_gpboost_mordred")


mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['text.usetex']     = False
mpl.rcParams['font.size']       = 24
mpl.rcParams['axes.titlesize']  = 24
mpl.rcParams['axes.labelsize']  = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.titlesize']= 24

def bootstrap_ci(group_df, value_col, n_boot=1000, alpha=0.05, random_state=123):
    """Compute mean and bootstrap CI over datasets within a group."""
    rng = np.random.default_rng(random_state)
    # average duplicates within dataset if present
    pivot = group_df.groupby('dataset')[value_col].mean()
    arr = pivot.values.astype(float)
    n   = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    point = float(np.nanmean(arr))
    if n == 1:  # with one dataset, CI = point
        return point, point, point
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.nanmean(arr[idx]))
    boots = np.asarray(boots, dtype=float)
    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi

def safe_unique(seq):
    s = []
    for x in seq:
        if x not in s:
            s.append(x)
    return s

df = pd.read_csv(BENCH_PATH) 

# Normalize names for robust filtering
df['model_norm']      = df['model'].str.lower()
df['fingerprint_norm']= df['fingerprint'].str.lower()
df['task']            = df['task'].str.lower()
df['split']           = df['split'].astype(str)

# Filter: GPBoost + Mordred only
sub = df[(df['model_norm'] == 'gpboost') & (df['fingerprint_norm'] == 'mordred')].copy()

if sub.empty:
    raise SystemExit("No rows found for model=GPBoost and fingerprint=Mordred. "
                     "Check casing/spelling in the CSV.")

# Target split order preference; fall back to whatever is present
preferred_order = ['random', 'group', 'scaffold']

splits_present  = safe_unique(sub['split'].str.lower().tolist())
split_order     = [s for s in preferred_order if s in splits_present] + \
                  [s for s in splits_present if s not in preferred_order]

# Aggregate with bootstrap CIs per task x split
rows = []
for split in split_order:
    for task in ['classification', 'regression']:
        g = sub[(sub['split'].str.lower() == split) & (sub['task'] == task)]
        if g.empty:
            rows.append({'split': split, 'task': task, 'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan})
            continue
        metric_col = 'f1' if task == 'classification' else 'rmse'
        mean, lo, hi = bootstrap_ci(g, metric_col)
        rows.append({'split': split, 'task': task, 'mean': mean, 'ci_low': lo, 'ci_high': hi})

agg = pd.DataFrame(rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

x = np.arange(len(split_order))

# Classification (F1, higher is better)
cls = agg[agg['task'] == 'classification'].set_index('split').reindex(split_order)
if not cls.empty:
    means = cls['mean'].values
    ci_lo = cls['ci_low'].values
    ci_hi = cls['ci_high'].values
    ax_top.bar(x, means, width=0.6)
    yerr = np.vstack([means - ci_lo, ci_hi - means])
    # Handle NaNs cleanly
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    ax_top.errorbar(x, means, yerr=yerr, fmt='none', capsize=4, linewidth=1)
    ax_top.set_ylabel('F1')
    ax_top.set_title('Classification')

# Regression (RMSE, lower is better)
reg = agg[agg['task'] == 'regression'].set_index('split').reindex(split_order)
if not reg.empty:
    means = reg['mean'].values
    ci_lo = reg['ci_low'].values
    ci_hi = reg['ci_high'].values
    ax_bot.bar(x, means, width=0.6)
    yerr = np.vstack([means - ci_lo, ci_hi - means])
    yerr = np.where(np.isnan(yerr), 0.0, yerr)
    ax_bot.errorbar(x, means, yerr=yerr, fmt='none', capsize=4, linewidth=1)
    ax_bot.set_ylabel('RMSE')
    ax_bot.set_title('Regression')

# X axis (splits)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(split_order)
ax_bot.set_xlabel('Split')

fig.tight_layout()
out_path = OUT_DIR / "gpboost_mordred_by_split.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)

print("Saved figure to:", out_path.resolve())
