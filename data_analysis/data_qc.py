# qc_variability_by_cas.py
import os, numpy as np, pandas as pd

IN_TRAIN = './benchmarking_datasets/origin_mordred_random_train.csv'
IN_TEST  = './benchmarking_datasets/origin_mordred_random_test.csv'
OUT_DIR  = './analysis_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

POWER10_TOL = 0.15   
IQR_FLAG    = 1.50  
RANGE_FLAG  = 3.00  
MIN_SPECIES = 3    

def safe_log10(x):
    try:
        v = float(x)
        return np.log10(v) if v > 0 else np.nan
    except:
        return np.nan

def main():
    df_tr = pd.read_csv(IN_TRAIN)
    df_te = pd.read_csv(IN_TEST)
    df = pd.concat([df_tr.assign(split='train'), df_te.assign(split='test')], ignore_index=True)

    df['y_log10'] = df['Effect value'].apply(safe_log10)
    df = df.dropna(subset=['y_log10']).copy()

    cas_col = 'CAS'
    species_col = 'Latin name'

    def per_cas_stats(g):
        y = g['y_log10'].values
        q1, q3 = np.quantile(y, [0.25, 0.75])
        iqr = q3 - q1
        rng = np.max(y) - np.min(y)
        mad = np.median(np.abs(y - np.median(y)))
        return pd.Series({
            'n_rows': len(g),
            'n_species': g[species_col].nunique(),
            'median_log10': np.median(y),
            'mean_log10': np.mean(y),
            'std_log10': np.std(y, ddof=1) if len(y) > 1 else 0.0,
            'iqr_log10': iqr,
            'range_log10': rng,
            'mad_log10': mad
        })

    cas_stats = df.groupby(cas_col).apply(per_cas_stats).reset_index()

    # flags
    cas_stats['flag_high_iqr']   = (cas_stats['n_species'] >= MIN_SPECIES) & (cas_stats['iqr_log10']  >= IQR_FLAG)
    cas_stats['flag_high_range'] = (cas_stats['n_species'] >= MIN_SPECIES) & (cas_stats['range_log10']>= RANGE_FLAG)
    cas_stats['flag_high_var']   = cas_stats[['flag_high_iqr','flag_high_range']].any(axis=1)

    cas_stats.sort_values(['flag_high_var', 'range_log10', 'iqr_log10'], ascending=[False, False, False]) \
             .to_csv(os.path.join(OUT_DIR, 'per_CAS_log10_variability.csv'), index=False)

    cas_median = cas_stats.set_index(cas_col)['median_log10'].to_dict()
    rows = []
    for i, r in df.iterrows():
        cas = r.get(cas_col)
        if pd.isna(cas): 
            continue
        med = cas_median.get(cas, np.nan)
        y   = r['y_log10']
        if pd.isna(med): 
            continue
        delta = y - med
        susp_k = 0
        for k in (1,2,3,4):
            if abs(abs(delta) - k) < POWER10_TOL:
                susp_k = int(np.sign(delta)*k)
                break
        rows.append({
            'CAS': cas,
            'Latin name': r.get(species_col, ''),
            'Effect value': r['Effect value'],
            'y_log10': y,
            'CAS_median_log10': med,
            'delta_log10': delta,
            'suspect_power10_k': susp_k
        })
    row_df = pd.DataFrame(rows)

    row_df.sort_values(['CAS','suspect_power10_k','delta_log10'], ascending=[True, False, False]) \
          .to_csv(os.path.join(OUT_DIR, 'row_level_vs_CAS_median.csv'), index=False)
    row_df[row_df['suspect_power10_k'] != 0] \
          .to_csv(os.path.join(OUT_DIR, 'suspect_power10_rows.csv'), index=False)

    top_var_cas = cas_stats[cas_stats['flag_high_var']].nlargest(50, 'range_log10')[cas_col].tolist()
    df_top = row_df[row_df['CAS'].isin(top_var_cas)]
    df_top.to_csv(os.path.join(OUT_DIR, 'top_variance_CAS_rows.csv'), index=False)

    print(f"[DONE] Wrote reports to {OUT_DIR}:")
    print("  - per_CAS_log10_variability.csv")
    print("  - row_level_vs_CAS_median.csv")
    print("  - suspect_power10_rows.csv")
    print("  - top_variance_CAS_rows.csv")

if __name__ == "__main__":
    main()
