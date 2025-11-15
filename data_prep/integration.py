import pandas as pd
import os
import re
import numpy as np

POWER10_TOL = 0.15  
DROP_K = {3, 4, -3, -4}

def drop_species_anchored_power10(df):
    d = df.copy()
    d['Effect value'] = pd.to_numeric(d['Effect value'], errors='coerce')
    d = d[d['Effect value'] > 0].copy()
    d['y_log10'] = np.log10(d['Effect value'].astype(float))

    key = ['CAS', 'Latin name', 'Duration (hours)']
    med = d.groupby(key)['y_log10'].median().rename('csd_median')
    d = d.merge(med, on=key, how='left')
    d['delta'] = d['y_log10'] - d['csd_median']

    def power10_k(delta):
        for k in (1, 2, 3, 4):
            if abs(abs(delta) - k) < POWER10_TOL:
                return int(np.sign(delta) * k)
        return 0

    d['suspect_power10_k'] = d['delta'].apply(power10_k)
    drop_mask = d['suspect_power10_k'].isin(DROP_K)

    dropped = d[drop_mask]
    kept    = d[~drop_mask]

    cols = df.columns
    return kept[cols], dropped[cols]

def detect_outliers(group):
    if len(group) < 5:
        return pd.DataFrame()

    y = group['Effect value'].astype(float)
    y = y[y > 0]                            
    if y.empty:
        return pd.DataFrame()

    ylog = np.log10(y)
    q1, q3 = np.quantile(ylog, [0.25, 0.75])
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    mask = (np.log10(group['Effect value']) < lb) | (np.log10(group['Effect value']) > ub)
    return group[mask]
def assign_synthetic_cas(df, start_id=1):
    df = df.copy()
    df['Canonical SMILES'] = df['Canonical SMILES'].astype(str).str.strip()
    
    missing_cas = df['CAS'].isna() | (df['CAS'].astype(str).str.strip() == '')
    
    unique_smiles = df.loc[missing_cas, 'Canonical SMILES'].dropna().unique()
    
    synthetic_map = {smiles: str(start_id + i) for i, smiles in enumerate(unique_smiles)}
    
    df.loc[missing_cas, 'CAS'] = df.loc[missing_cas, 'Canonical SMILES'].map(synthetic_map)
    
    return df


def cross_species_supported(row, df):
        if not row['extreme']:
            return True
        if pd.isna(row['Taxonomic order']):
            return False
        same_order = df[
            (df['CAS'] == row['CAS']) &
            (df['Taxonomic order'] == row['Taxonomic order']) &
            (df['Latin name'] != row['Latin name']) &
            (df['Effect value'] >= 1e-5) & (df['Effect value'] <= 1e5)
        ]
        return not same_order.empty

def first_non_null(series):
    return series.dropna().iloc[0] if not series.dropna().empty else None

def build_canonical_taxonomy(processed_files, processed_files_dir, envirotox_file):
    taxonomy_columns = [
        'Latin name', 'Taxonomic kingdom',
        'Taxonomic phylum or division', 'Taxonomic subphylum',
        'Taxonomic class', 'Taxonomic order', 'Taxonomic family'
    ]
    
    all_taxonomy_rows = []

    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_dir, file))


        for col in taxonomy_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        tax_rows = df[taxonomy_columns].dropna(subset=['Latin name'])
        all_taxonomy_rows.append(tax_rows)

    taxonomy_env = pd.read_excel(envirotox_file, sheet_name='taxonomy')
    def clean_taxonomy_entry(entry):
        if pd.isna(entry):
            return None
        return re.sub(r'\s*\(.*?\)', '', str(entry)).strip().lower()

    for col in taxonomy_columns:
        taxonomy_env[col] = taxonomy_env[col].apply(clean_taxonomy_entry)
    all_taxonomy_rows.append(taxonomy_env[taxonomy_columns])

    combined = pd.concat(all_taxonomy_rows, axis=0, ignore_index=True)
    canonical_taxonomy = combined.drop_duplicates(subset=['Latin name'], keep='last')
    for col in taxonomy_columns:
        canonical_taxonomy[col] = canonical_taxonomy[col].apply(clean_taxonomy_entry)
    return canonical_taxonomy

def integrate_and_process_datasets(file_paths, output_file):
    dataframes = []

    envirotox_file = './raw_data/envirotox.xlsx'
    canonical_taxonomy = build_canonical_taxonomy(file_paths, processed_files_dir, envirotox_file)
    canonical_taxonomy.to_csv('./datasets/species_taxonomy_origin.csv', index=False)
    common_to_latin = {
            'bluegill': 'lepomis macrochirus',
            'eastern oyster': 'crassostrea virginica',
            'rainbow trout': 'oncorhynchus mykiss',
            'sheepshead minnow': 'cyprinodon variegatus',
            'zebra fish': 'danio rerio',
            'zebrafish': 'danio rerio' 
        }
    for file in file_paths:
        file_path = os.path.join(processed_files_dir, file)
        df = pd.read_csv(file_path)

        df['Original CAS'] = df['CAS']
        df['CAS'] = df['CAS'].astype(str).apply(lambda x: re.sub(r'\D', '', x))

        df.rename(columns={'Effect value (mg/L)': 'Effect value'}, inplace=True)

        df['Latin name'] = df['Latin name'].astype(str).str.strip().str.lower()


        df['Latin name'] = df['Latin name'].replace(common_to_latin)

        taxonomy_fields = [col for col in canonical_taxonomy.columns if col != 'Latin name']
        df.drop(columns=[col for col in taxonomy_fields if col in df.columns], inplace=True)
        df = df.merge(canonical_taxonomy, on='Latin name', how='left')

        dataframes.append(df)

    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    combined_df = assign_synthetic_cas(combined_df)


    # combined_df['Effect value'] = pd.to_numeric(combined_df['Effect value'], errors='coerce')
    # combined_df.dropna(subset=['Effect value'], inplace=True)

    # grouped = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)'])
    # outliers = grouped.apply(detect_outliers).reset_index(drop=True)
    # print("Outliers detected and removed:", len(outliers))
    # combined_df = combined_df[~combined_df.index.isin(outliers.index)]

    # grouped_stats = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)'])['Effect value'].agg(['mean', 'std']).reset_index()
    # grouped_stats.rename(columns={'mean': 'Effect value', 'std': 'Effect value std'}, inplace=True)

    # other_columns = [col for col in combined_df.columns if col not in ['CAS', 'Latin name', 'Duration (hours)', 'Effect value']]
    # aggregation = {col: first_non_null for col in other_columns}
    # meta_df = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)']).agg(aggregation).reset_index()

    # integrated_df = pd.merge(grouped_stats, meta_df, on=['CAS', 'Latin name', 'Duration (hours)'])


    # integrated_df.to_csv(output_file, index=False)
    # print(f"Integrated dataset saved to: {output_file}")
    combined_df['Effect value'] = pd.to_numeric(combined_df['Effect value'], errors='coerce')
    combined_df.dropna(subset=['Effect value'], inplace=True)

    # Mark extreme values
    combined_df['extreme'] = (combined_df['Effect value'] < 1e-5) | (combined_df['Effect value'] > 1e5)

    

    # Apply support check and store result
    combined_df['has_support'] = combined_df.apply(lambda row: cross_species_supported(row, combined_df), axis=1)
    support_count = combined_df['has_support'].sum()
    total_extreme = combined_df['extreme'].sum()
    print(f"{support_count} out of {total_extreme} extreme-value samples are supported by cross-species evidence.")

    # Filter out unsupported extreme values
    combined_df = combined_df[(~combined_df['extreme']) | (combined_df['has_support'])]
    combined_df.drop(columns=['extreme', 'has_support'], inplace=True)
    
    combined_df, dropped_power10 = drop_species_anchored_power10(combined_df)
    dropped_power10.to_csv('./analysis_outputs/dropped_species_power10_pm34.csv', index=False)


    grouped = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)'])
    outliers = grouped.apply(detect_outliers).reset_index(drop=True)
    print("Outliers removed:", len(outliers))
    combined_df = combined_df[~combined_df.index.isin(outliers.index)]

    # Aggregate Effect value and remove high std/mean rows 
    grouped_stats = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)'])['Effect value'].agg(['mean', 'std']).reset_index()
    grouped_stats.rename(columns={'mean': 'Effect value', 'std': 'Effect value std'}, inplace=True)

    # Filter based on std/mean threshold
    grouped_stats = grouped_stats[
    (grouped_stats['Effect value std'].isna()) |
    (grouped_stats['Effect value std'] <= 1.5 * grouped_stats['Effect value'])]

    # Merge with metadata 
    other_columns = [col for col in combined_df.columns if col not in ['CAS', 'Latin name', 'Duration (hours)', 'Effect value']]
    aggregation = {col: first_non_null for col in other_columns}
    meta_df = combined_df.groupby(['CAS', 'Latin name', 'Duration (hours)']).agg(aggregation).reset_index()

    # Merge final
    integrated_df = pd.merge(grouped_stats, meta_df, on=['CAS', 'Latin name', 'Duration (hours)'])
    integrated_df.to_csv(output_file, index=False)


processed_files_dir = './processed_files/'
processed_files = [f for f in os.listdir(processed_files_dir) if f.endswith('.csv')]
print("Processed files found:", processed_files)

output_file = './datasets/integrated_dataset_log10detect.csv'
integrate_and_process_datasets(processed_files, output_file)
