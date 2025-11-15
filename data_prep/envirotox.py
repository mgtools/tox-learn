import pandas as pd

def combine_and_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows with the same 'CAS' and 'Latin name' and average the 'Effect value',
    while retaining other specified columns.
    """
    # Combine and average 'Effect value' for duplicate 'CAS', 'Latin name', and 'Duration (hours)'
    aggregation = {col: 'first' for col in df.columns if col not in ['CAS', 'Latin name', 'Duration (hours)', 'Effect value']}
    aggregation['Effect value'] = 'mean'
    df = df.groupby(['CAS', 'Latin name', 'Duration (hours)']).agg(aggregation).reset_index()
    
    return df

def enviro_tox_preprocess(file_path: str, output_path: str):
    """
    Preprocess the EnviroTox dataset by filtering data, merging with 'substance' and 'taxonomy' sheets,
    and retaining specified columns.
    """
    # Load the main dataset
    enviro_tox = pd.read_excel(file_path, sheet_name='test')
    
    # Filter the data based on specified conditions
    filtered_data = enviro_tox[
        (enviro_tox['Test statistic'].isin(['LC50', 'EC50'])) &
        (enviro_tox['Duration (hours)'].isin([24, 48, 72, 96])) &
        (enviro_tox['Effect is 5X above water solubility'] == 0)
    ][['CAS', 'Chemical name', 'Latin name', 'Trophic Level', 'Effect value', 'Test statistic', 'Duration (hours)']]
     
    # Load the 'substance' sheet to get 'Canonical SMILES'
    substance = pd.read_excel(file_path, sheet_name='substance')
    substance = substance[['original CAS', 'Canonical SMILES', 'Water Solubility (mg/L)']]
    # Standardize column names
    substance.rename(columns={'Water Solubility (mg/L)': 'Water solubility'}, inplace=True)

    # Merge the filtered data with the 'substance' sheet
    merged_df = pd.merge(filtered_data, substance, left_on='CAS', right_on='original CAS', how='left')
    merged_df.drop(columns=['original CAS'], inplace=True)
    
    # Load the 'taxonomy' sheet to get taxonomic information
    taxonomy = pd.read_excel(file_path, sheet_name='taxonomy')
    taxonomy_columns = [
        'Latin name', 'Trophic Level', 'Medium', 'Taxonomic kingdom', 'Taxonomic phylum or division',
        'Taxonomic subphylum', 'Taxonomic superclass', 'Taxonomic class', 'Taxonomic order', 'Taxonomic family'
    ]
    taxonomy = taxonomy[taxonomy_columns]
    
    # Merge the data with the 'taxonomy' sheet
    final_df = pd.merge(merged_df, taxonomy, on=['Latin name', 'Trophic Level'], how='left')
    
    final_df.dropna(subset=['Canonical SMILES'], inplace=True)

    
    # Save the processed data to a CSV file
    final_df.to_csv(output_path, index=False)

enviro_tox_preprocess('./raw_data/envirotox.xlsx', './processed_files/envirotox_processed.csv')
