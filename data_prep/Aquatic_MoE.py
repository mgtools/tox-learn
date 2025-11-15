import pandas as pd
import numpy as np
def detect_outliers(group):
    Q1 = group['Effect value'].quantile(0.25)
    Q3 = group['Effect value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 10 * IQR
    upper_bound = Q3 + 10 * IQR
    outliers = group[(group['Effect value'] < lower_bound) | (group['Effect value'] > upper_bound)]
    return outliers

def process_dataset(input_file: str, ws_file: str, output_file: str):
    """
    Process the dataset to align with the EnviroTox structure, incorporating water solubility data.

    Parameters:
    - input_file: Path to the input CSV file.
    - ws_file: Path to the water solubility CSV file.
    - output_file: Path to save the processed CSV file.
    """

    # Load the main dataset
    dtype_spec = {'CAS Number': str}
    df = pd.read_csv(input_file, encoding='utf-8', dtype=dtype_spec)

    # Load the water solubility dataset
    ws_df = pd.read_csv(ws_file, dtype=dtype_spec)

    # Filter water solubility data for relevant units
    ws_df = ws_df[ws_df['Value.Unit'].isin(['mg/L', 'mg/kg'])]

    # Convert all water solubility values to mg/L
    ws_df['Water Solubility (mg/L)'] = ws_df.apply(
        lambda row: row['Value.MeanValue'] ,
        axis=1
    )

    # Aggregate multiple solubility values per chemical by their mean
    ws_agg = ws_df.groupby('CAS Number')['Water Solubility (mg/L)'].mean().reset_index()

    # Merge the water solubility data with the main dataset
    df = pd.merge(df, ws_agg, on='CAS Number', how='left')

    # Standardize 'CAS Number' by replacing '/' with '-'
    df['CAS Number'] = df['CAS Number'].str.replace('/', '-')

    # Define the column mapping to align with EnviroTox
    column_mapping = {
        'CAS Number': 'CAS',
        'SMILES': 'Canonical SMILES',
        'Test organisms (species)': 'Latin name',
        'Endpoint': 'Test statistic',
        'Duration.MeanValue': 'Duration (hours)',
        'Value.MeanValue': 'Effect value',
        'Kingdom': 'Taxonomic kingdom',
        'Phylum': 'Taxonomic phylum or division',
        'Subphylum': 'Taxonomic subphylum',
        'Class': 'Taxonomic class',
        'Order': 'Taxonomic order',
        'Family': 'Taxonomic family',
        'Medium': 'Water type',
        'Water Solubility (mg/L)': 'Water solubility'
    }

    df.rename(columns=column_mapping, inplace=True)

    # Select relevant columns
    relevant_columns = [
        'CAS', 'Latin name', 'Effect value',
        'Test statistic', 'Duration (hours)', 'Canonical SMILES',
        'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
        'Taxonomic class', 'Taxonomic order', 'Taxonomic family', 'Water solubility'
    ]
    df = df[relevant_columns]

    # Filter based on specific conditions
    df = df[
        (df['Test statistic'].isin(['LC50', 'EC50'])) &
        (df['Duration (hours)'].isin([24, 48, 72, 96]))
    ]

    # Drop rows where 'Canonical SMILES' is NaN
    df.dropna(subset=['Canonical SMILES'], inplace=True)

    # Convert 'Effect value' to numeric
    df['Effect value'] = pd.to_numeric(df['Effect value'], errors='coerce')

    # Drop rows where 'Effect value' is NaN or infinite
    df = df.dropna(subset=['Effect value'])
    df = df[np.isfinite(df['Effect value'])]

    # Filter based on water solubility
    df = df[
        (df['Effect value'] <= 5 * df['Water solubility']) |
        (df['Water solubility'].isna() & (df['Effect value'] <= 1e5) & (df['Effect value'] >= 1e-5))
    ]



    # Save the processed DataFrame to a CSV file
    df.to_csv(output_file, index=False)

input_file = './raw_data/Aquatic Japan MoE-LC50-EC50.csv'
output_file = './processed_files/processed_Aquatic_Japan_MoE.csv'
ws_file = './raw_data/Aquatic Japan MoE-WS.csv'
process_dataset(input_file, ws_file, output_file)
