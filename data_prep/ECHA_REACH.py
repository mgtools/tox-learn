from xdrlib import ConversionError
import pandas as pd
import numpy as np
import re


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
    df = pd.read_csv(input_file, dtype=dtype_spec)

    # Load the water solubility dataset
    ws_df = pd.read_csv(ws_file, dtype=dtype_spec)

    # Filter water solubility data for relevant units
    ws_df = ws_df[ws_df['Value.Unit'].isin(['mg/L', 'mg/kg'])]

    # Convert all water solubility values to mg/L
    ws_df['Water Solubility (mg/L)'] = ws_df.apply(
        lambda row: row['Value.MeanValue'],
        axis=1
    )

    # Aggregate multiple solubility values per chemical by their mean
    ws_agg = ws_df.groupby('CAS Number')['Water Solubility (mg/L)'].mean().reset_index()

    # Merge the water solubility data with the main dataset
    df = pd.merge(df, ws_agg, on='CAS Number', how='left')

    # Filter out rows where 'CAS Number' is 'No CAS number'
    df = df[df['CAS Number'] != 'No CAS number']
    df = df[df['Test organisms (species)'] != 'Other Test organisms (species)']
    # Standardize 'CAS Number' by replacing '/' with '-'
    df['CAS Number'] = df['CAS Number'].str.replace('/', '-')
    
    if 'Duration.Unit' in df.columns:
        df['Duration.MeanValue'] = df.apply(
            lambda row: row['Duration.MeanValue'] * 24 if str(row['Duration.Unit']).strip().lower() in ['day', 'days'] else row['Duration.MeanValue'],
            axis=1
        )    

    # Define the column mapping to align with EnviroTox
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
        'Water media type': 'Water type',
        'Water Solubility (mg/L)': 'Water solubility'
    }
    unit_patterns = {
        r'\bmg/?l\b': 1,
        r'\bppm\b': 1,
        r'\bmg\b.*?/l': 1,
        r'\bug\b.*?/l': 0.001,
        r'\bng\b.*?/l': 1e-6,
        r'\bppb\b': 0.001,
        r'\bg\b.*?/l': 1000,
    }
    def get_conversion_factor(unit):
        if pd.isna(unit):
            return None
        unit = unit.strip().lower()
        for pattern, factor in unit_patterns.items():
            if re.search(pattern, unit):
                return factor
        return None  # Unknown unit

    df.rename(columns=column_mapping, inplace=True)

    # Select relevant columns
    relevant_columns = [
        'CAS', 'Latin name', 'Effect value', 'Effect value unit',
        'Test statistic', 'Duration (hours)', 'Canonical SMILES', 'Water type',
        'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
        'Taxonomic class', 'Taxonomic order', 'Taxonomic family', 'Water solubility'
    ]
    df = df[relevant_columns]

    # Filter based on specific conditions
    df = df[
        (df['Test statistic'].isin(['LC50', 'EC50'])) &
        (df['Duration (hours)'].isin([24, 48, 72, 96]))
    ]

    df.dropna(subset=['Canonical SMILES'], inplace=True)


    df['conversion_factor'] = df['Effect value unit'].apply(get_conversion_factor)

    df = df[df['conversion_factor'].notna()]

    df['Effect value'] = pd.to_numeric(df['Effect value'], errors='coerce')
    df['Effect value (mg/L)'] = df['Effect value'] * df['conversion_factor']


    # Drop rows where 'Effect value (mg/L)' is NaN or infinite
    df = df.dropna(subset=['Effect value (mg/L)'])
    df = df[np.isfinite(df['Effect value (mg/L)'])]

    # Drop the original 'Effect value' and 'Effect value unit' columns
    df.drop(columns=['Effect value', 'conversion_factor', 'Effect value unit'], inplace=True)
    
    # Filter based on water solubility
    df = df[
        (df['Effect value (mg/L)'] <= 5 * df['Water solubility']) |
        (df['Water solubility'].isna() & (df['Effect value (mg/L)'] <= 1e5) & (df['Effect value (mg/L)'] >= 1e-5))
    ]
    

    # Save the processed DataFrame to a CSV file
    df.to_csv(output_file, index=False)

input_file = './raw_data/ECHA REACH-LC50-EC50-2.csv'
output_file = './processed_files/processed_ECHA-2.csv'
ws_file = './raw_data/ECHA REACH-WS.csv'
process_dataset(input_file, ws_file, output_file) 
