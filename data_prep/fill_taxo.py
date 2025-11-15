from ete3 import NCBITaxa
import pandas as pd
import re
import numpy as np

ncbi = NCBITaxa()


def strip_brackets(entry):
    if pd.isna(entry):
        return None
    return re.sub(r'\s*[\(\[].*?[\)\]]', '', str(entry)).strip().lower()

def is_valid_latin_name(name):
    if pd.isna(name):
        return False
    parts = str(name).strip().split()
    return len(parts) == 2 and all(part.isalpha() for part in parts)

def get_taxonomy_from_ncbi(latin_name):
    try:
        taxid_dict = ncbi.get_name_translator([latin_name])
        if not taxid_dict:
            return {}
        taxid = list(taxid_dict.values())[0][0]
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage)
        ranks = ncbi.get_rank(lineage)

        tax_map = {}
        for taxid in lineage:
            rank = ranks[taxid]
            name = names[taxid].lower()
            if rank == "kingdom":
                tax_map["Taxonomic kingdom"] = name
            elif rank == "phylum":
                tax_map["Taxonomic phylum or division"] = name
            elif rank == "subphylum":
                tax_map["Taxonomic subphylum"] = name
            elif rank == "class":
                tax_map["Taxonomic class"] = name
            elif rank == "order":
                tax_map["Taxonomic order"] = name
            elif rank == "family":
                tax_map["Taxonomic family"] = name
        return tax_map
    except:
        return {}

def replace_taxonomy_with_ncbi(df):
    taxonomy_cols = [
        'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
        'Taxonomic class', 'Taxonomic order', 'Taxonomic family'
    ]

    df = df.drop(columns=[col for col in taxonomy_cols if col in df.columns])

    for col in taxonomy_cols:
        df[col] = None

    for idx, row in df.iterrows():
        latin_name = row['Latin name']
        if is_valid_latin_name(latin_name):
            ncbi_data = get_taxonomy_from_ncbi(latin_name)
            for col, val in ncbi_data.items():
                df.at[idx, col] = val

    return df

def fill_taxonomy_hierarchy(df, taxonomy_columns, superclass_column='Taxonomic superclass'):
    df = df.copy()

    for i in range(len(taxonomy_columns) - 1, 0, -1):
        lower, higher = taxonomy_columns[i], taxonomy_columns[i - 1]
        df[lower] = df[lower].fillna(df[higher].apply(lambda x: f"Unknown_from_{higher}_{x}" if pd.notna(x) else np.nan))

    for i in range(len(taxonomy_columns) - 1):
        higher, lower = taxonomy_columns[i], taxonomy_columns[i + 1]
        df[higher] = df[higher].fillna(df[lower].apply(lambda x: f"Unknown_from_{lower}_{x}" if pd.notna(x) else np.nan))

    def fill_from_superclass(row, col):
        if pd.notna(row[col]):
            return row[col]
        superclass = row.get(superclass_column, "")
        if pd.isna(superclass):
            return f"Unknown_{col}"
        if superclass.lower() in ["actinopteri"]:
            return f"Unknown_{col}_fish"
        elif superclass.lower() in ["malacostraca", "branchiopoda", "hexanauplia"]:
            return f"Unknown_{col}_invertebrate"
        elif superclass.lower() in ["amphibia"]:
            return f"Unknown_{col}_amphibian"
        else:
            return f"Unknown_{col}_{superclass}"

    for col in taxonomy_columns:
        df[col] = df.apply(lambda row: fill_from_superclass(row, col), axis=1)

    return df


file = './datasets/integrated_dataset.csv'
df = pd.read_csv(file)

df = replace_taxonomy_with_ncbi(df)

taxonomy_columns = [
    'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
    'Taxonomic class', 'Taxonomic order', 'Taxonomic family'
]
for col in taxonomy_columns:
    df[col] = df[col].apply(strip_brackets)
    
taxonomy_metadata = df[['Latin name'] + taxonomy_columns].drop_duplicates(subset=['Latin name'])

taxonomy_metadata.to_csv('./datasets/species_taxonomy_ncbi.csv', index=False)

df.to_csv('./datasets/integrated_dataset_ncbi.csv', index=False)

df_orig = pd.read_csv('./datasets/integrated_dataset_log10detect.csv')
df_ncbi = pd.read_csv('./datasets/integrated_dataset_ncbi.csv')

taxonomy_columns = [
    'Taxonomic kingdom', 'Taxonomic phylum or division', 'Taxonomic subphylum',
    'Taxonomic class', 'Taxonomic order', 'Taxonomic family'
]

df_orig_clean = df_orig.dropna(subset=taxonomy_columns, how='all')
df_ncbi_clean = df_ncbi.dropna(subset=taxonomy_columns, how='all')

df_orig_filled = fill_taxonomy_hierarchy(df_orig_clean, taxonomy_columns)
df_ncbi_filled = fill_taxonomy_hierarchy(df_ncbi_clean, taxonomy_columns)

df_orig_filled.to_csv('./datasets/integrated_dataset_log10detect_filled.csv', index=False)
df_ncbi_filled.to_csv('./datasets/integrated_dataset_ncbi_filled.csv', index=False)

common_keys = pd.merge(
    df_orig_filled[['CAS', 'Canonical SMILES', 'Latin name', 'Duration (hours)']],
    df_ncbi_filled[['CAS', 'Canonical SMILES', 'Latin name', 'Duration (hours)']],
    on=['CAS', 'Canonical SMILES', 'Latin name', 'Duration (hours)'],
    how='inner'
).drop_duplicates()

df_bench_orig = pd.merge(df_orig_filled, common_keys, on=['CAS', 'Canonical SMILES', 'Latin name', 'Duration (hours)'])
df_bench_ncbi = pd.merge(df_ncbi_filled, common_keys, on=['CAS', 'Canonical SMILES', 'Latin name', 'Duration (hours)'])

df_bench_orig.to_csv('./datasets/integrated_dataset_origin_bench.csv', index=False)
df_bench_ncbi.to_csv('./datasets/integrated_dataset_ncbi_bench.csv', index=False)

print(f"Saved benchmark-aligned datasets: {len(df_bench_orig)} samples")