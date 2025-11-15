from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np
import os

def calculate_mordred_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        calc = Calculator(descriptors, ignore_3D=True)
        result = calc(mol)
        descriptor_values = list(result.values())
        return descriptor_values
    except Exception:
        return [-1] * 1613  # Number of Mordred descriptors

def build_chemical_fingerprint(df, output_path):
    unique_chemicals = df[['CAS', 'Canonical SMILES']].drop_duplicates()
    
    calc = Calculator(descriptors, ignore_3D=True)
    descriptor_names = [str(d) for d in calc.descriptors]

    fingerprints = []
    failed = []

    for idx, row in unique_chemicals.iterrows():
        cas = row['CAS']
        smiles = row['Canonical SMILES']
        values = calculate_mordred_descriptors(smiles)
        if values.count(-1) != len(values):
            fingerprints.append([cas, smiles] + values)
        else:
            failed.append((cas, smiles))

    fingerprint_df = pd.DataFrame(fingerprints, columns=['CAS', 'Canonical SMILES'] + descriptor_names)
    fingerprint_df.to_csv(output_path, index=False)
    print(f"Saved fingerprints to {output_path}. Failed entries: {len(failed)}")
    return fingerprint_df

def merge_fingerprints(integrated_file, fingerprint_file, output_file):
    df = pd.read_csv(integrated_file)
    fp_df = pd.read_csv(fingerprint_file)

    # Merge by CAS and SMILES
    df = df.merge(fp_df, on=['CAS', 'Canonical SMILES'], how='left')

    df.to_csv(output_file, index=False)
    print(f"Saved merged dataset to {output_file}")
    return df

def clean_mordred_fingerprints(file_path, output_path, nan_col_threshold=0.2):
    df = pd.read_csv(file_path)

    # Convert all non-numeric values (except CAS/SMILES) to NaN
    for col in df.columns[2:]:  # Keep CAS, SMILES
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that have all NaN in descriptor columns
    df = df.dropna(axis=0, how='all', subset=df.columns[2:])

    # Drop descriptor columns with too many missing values
    threshold = int((1 - nan_col_threshold) * len(df))
    df = df.dropna(axis=1, thresh=threshold)

    # Fill remaining NaNs (those descriptors missing in a few compounds) with 0
    df[df.columns[2:]] = df[df.columns[2:]].fillna(0)

    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"Cleaned Mordred file saved to {output_path} with shape: {df.shape}")
    return df

integrated_path = './datasets/integrated_dataset_log10detect_filled.csv'
fingerprint_path = './datasets/chemical_fingerprints_mordred.csv'
# output_path = './datasets/integrated_dataset_with_fingerprints.csv'

df_integrated = pd.read_csv(integrated_path)
# fp_df = build_chemical_fingerprint(df_integrated, fingerprint_path)

clean_mordred_fingerprints(
    file_path='./datasets/chemical_fingerprints_mordred.csv',
    output_path='./datasets/chemical_fingerprints_mordred_clean_0.csv',
    nan_col_threshold=0.1  # drop columns with >10% NaN
)

# merged_df = merge_fingerprints(integrated_path, fingerprint_path, output_path)
