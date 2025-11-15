import os, numpy as np, pandas as pd

IN_TRAIN = './benchmarking_datasets/origin_mordred_random_train.csv'
IN_TEST  = './benchmarking_datasets/origin_mordred_random_test.csv'
OUT_DIR  = './analysis_outputs'
OUT_TRAIN = './benchmarking_datasets/origin_mordred_random_train_clean.csv'
OUT_TEST  = './benchmarking_datasets/origin_mordred_random_test_clean.csv'

POWER10_TOL = 0.15   
DROP_K = {3, 4, -3, -4} 
CAS_COL = 'CAS'
SPECIES_COL = 'Latin name'
VALUE_COL = 'Effect value'

os.makedirs(OUT_DIR, exist_ok=True)

def safe_log10(x):
    try:
        v = float(x)
        return np.log10(v) if v > 0 else np.nan
    except:
        return np.nan

def main():
    df_tr = pd.read_csv(IN_TRAIN)
    df_te = pd.read_csv(IN_TEST)

    df_tr['__split__'] = 'train'
    df_te['__split__'] = 'test'
    df = pd.concat([df_tr, df_te], ignore_index=True)

    # compute log10 LC50
    df['y_log10'] = df[VALUE_COL].apply(safe_log10)
    before_len = len(df)
    df = df.dropna(subset=['y_log10', CAS_COL]).copy()

    # CAS medians in log10 across ALL rows (train+test) so flags are consistent
    cas_median = df.groupby(CAS_COL)['y_log10'].median()

    def power10_k(row):
        med = cas_median.get(row[CAS_COL], np.nan)
        y = row['y_log10']
        if np.isnan(med) or np.isnan(y): 
            return 0
        delta = y - med
        for k in (1, 2, 3, 4):
            if abs(abs(delta) - k) < POWER10_TOL:
                return int(np.sign(delta) * k)
        return 0

    df['suspect_power10_k'] = df.apply(power10_k, axis=1)

    # Save all suspects
    suspects = df[df['suspect_power10_k'] != 0].copy()
    suspects.to_csv(os.path.join(OUT_DIR, 'suspect_power10_rows_all.csv'), index=False)

    drop_mask = df['suspect_power10_k'].isin(DROP_K)
    dropped = df[drop_mask].copy()
    kept = df[~drop_mask].copy()

    summary = {
        'total_rows_input'     : int(before_len),
        'rows_with_valid_log10': int(len(df)),
        'suspects_all'         : int(len(suspects)),
        'dropped_rows'         : int(len(dropped)),
        'kept_rows'            : int(len(kept)),
    }
    pd.Series(summary).to_csv(os.path.join(OUT_DIR, 'clean_drop_power10_summary.csv'))

    dropped.to_csv(os.path.join(OUT_DIR, 'dropped_power10_k.csv'), index=False)

    tr_clean = kept[kept['__split__'] == 'train'].drop(columns=['__split__'])
    te_clean = kept[kept['__split__'] == 'test'].drop(columns=['__split__'])

    tr_clean.to_csv(OUT_TRAIN, index=False)
    te_clean.to_csv(OUT_TEST, index=False)

    print("[CLEAN] Done.")
    print(f"  Input total               : {before_len}")
    print(f"  With valid log10          : {len(df)}")
    print(f"  Suspects (any)     : {len(suspects)}")
    print(f"  Dropped (k in {sorted(DROP_K)}) : {len(dropped)}")
    print(f"  Kept                      : {len(kept)}")
    print(f"  Wrote cleaned train  {OUT_TRAIN}")
    print(f"  Wrote cleaned test   {OUT_TEST}")
    print(f"  Audit files in {OUT_DIR}")
    
if __name__ == "__main__":
    main()
