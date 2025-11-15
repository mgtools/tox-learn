import pandas as pd
import numpy as np

def process_dataset(input_file: str, ws_file: str, output_file: str):
    """
    Process the dataset using ADORE-inspired preprocessing.
    """
    dtype_spec = {'CAS Number': str}
    df = pd.read_csv(input_file, dtype=dtype_spec)
    ws_df = pd.read_csv(ws_file, dtype=dtype_spec)

    #  Fill in missing Value.MeanValue by averaging Min and Max
    df['Value.MeanValue'] = df.apply(
        lambda row: np.mean([row['Value.MinValue'], row['Value.MaxValue']])
        if pd.isna(row['Value.MeanValue']) and not pd.isna(row['Value.MinValue']) and not pd.isna(row['Value.MaxValue'])
        else row['Value.MeanValue'],
        axis=1
    )

    # Filter unrealistic pH values if any pH is available
    def filter_unrealistic_ph(row):
        phs = [row.get(k) for k in ['Media ph.MinValue', 'Media ph.MaxValue', 'Media ph.MeanValue']]
        phs = [p for p in phs if pd.notna(p)]
        if phs and (min(phs) < 4 or max(phs) > 10):
            return False
        return True
    df = df[df.apply(filter_unrealistic_ph, axis=1)]

    # Process water solubility
    ws_df = ws_df[ws_df['Value.Unit'].isin(['mg/L', 'mg/kg'])]
    ws_df['Water Solubility (mg/L)'] = ws_df['Value.MeanValue']
    ws_agg = ws_df.groupby('CAS Number')['Water Solubility (mg/L)'].mean().reset_index()
    df = pd.merge(df, ws_agg, on='CAS Number', how='left')

    # Clean CAS numbers
    df = df[df['CAS Number'] != 'No CAS number']
    df['CAS Number'] = df['CAS Number'].str.replace('/', '-')
    
    if 'Duration.Unit' in df.columns:
        df['Duration.MeanValue'] = df.apply(
            lambda row: row['Duration.MeanValue'] * 24 if str(row['Duration.Unit']).strip().lower() in ['day', 'days'] else row['Duration.MeanValue'],
            axis=1
        )

    # Rename columns
    column_mapping = {
        'CAS Number': 'CAS',
        'SMILES': 'Canonical SMILES',
        'Test organisms (species)': 'Latin name',
        'Endpoint': 'Test statistic',
        'Duration.MeanValue': 'Duration (hours)',
        'Value.MeanValue': 'Effect value',
        'Value.Unit': 'Effect value unit',
        'Kingdom': 'Taxonomic kingdom',
        'Phylum': 'Taxonomic phylum or division',
        'Subphylum': 'Taxonomic subphylum',
        'Class': 'Taxonomic class',
        'Order': 'Taxonomic order',
        'Family': 'Taxonomic family',
        'Media type': 'Water type',
        'Water Solubility (mg/L)': 'Water solubility'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Retain relevant columns
    relevant_columns = [
        'CAS', 'Latin name', 'Effect value', 'Effect value unit',
        'Test statistic', 'Duration (hours)', 'Canonical SMILES', 'Water type',
        'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
        'Taxonomic class', 'Taxonomic order', 'Taxonomic family', 'Water solubility'
    ]
    df = df[relevant_columns]

    # Filter EC50/LC50 + duration
    df = df[
        (df['Test statistic'].isin(['LC50', 'EC50'])) &
        (df['Duration (hours)'].isin([24, 48, 72, 96]))
    ]

    # Drop rows with missing SMILES or invalid units
    df.dropna(subset=['Canonical SMILES'], inplace=True)
    valid_units = {
        'ppm': 1,
        'mg/L': 1,
        'ug/L': 0.001,
        'ng/L': 0.000001,
        'ppt': 0.001,
        'ppb': 0.001,
        'AI ppm': 1,
        'ae mg/L': 1,
        'ae ug/L': 0.001,
        'ae ppm': 1,
        'AI ppb': 0.001,
        'mg/kg': 1,
    }
    df = df[df['Effect value unit'].isin(valid_units.keys())]

    # Convert effect values to mg/L
    df['Effect value'] = pd.to_numeric(df['Effect value'], errors='coerce')
    df['Effect value (mg/L)'] = df.apply(
        lambda row: row['Effect value'] * valid_units[row['Effect value unit']],
        axis=1
    )
    df.drop(columns=['Effect value', 'Effect value unit'], inplace=True)
    df = df.dropna(subset=['Effect value (mg/L)'])
    df = df[np.isfinite(df['Effect value (mg/L)'])]

    # Filter based on water solubility or global bounds if solubility is missing
    df = df[
        (df['Effect value (mg/L)'] <= 5 * df['Water solubility']) |
        (df['Water solubility'].isna() & (df['Effect value (mg/L)'] <= 1e5) & (df['Effect value (mg/L)'] >= 1e-5))
    ]

    # Save the result
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to: {output_file}")

input_file = './raw_data/ECOTOX-LC50-EC50-2.csv'
ws_file = './raw_data/ECOTOX-WS.csv'
output_file = './processed_files/processed_ECOTOX.csv'

process_dataset(input_file, ws_file, output_file)
