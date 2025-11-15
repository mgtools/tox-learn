from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
import pandas as pd

def compute_rdkit_fingerprint(smiles, fingerprint_type="Morgan", nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if fingerprint_type == "Morgan":
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits))
        elif fingerprint_type == "MACCS":
            return list(MACCSkeys.GenMACCSKeys(mol))
        elif fingerprint_type == "RDKit":
            return list(Chem.RDKFingerprint(mol, fpSize=nBits))
        elif fingerprint_type == "AtomPair":
            return list(Pairs.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits))
        elif fingerprint_type == "Torsion":
            return list(Torsions.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits))
        else:
            raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")

    except Exception as e:
        print(f"[Error] {smiles}: {e}")
        return None

def build_rdkit_fingerprint_file(df, output_path, fingerprint_type="Morgan", nBits=2048):
    assert fingerprint_type in ["Morgan", "MACCS", "RDKit", "AtomPair", "Torsion"], "Invalid fingerprint type"

    unique_chemicals = df[['CAS', 'Canonical SMILES']].drop_duplicates()
    fingerprints = []
    failed = []

    for idx, row in unique_chemicals.iterrows():
        cas = row['CAS']
        smiles = row['Canonical SMILES']
        
        values = compute_rdkit_fingerprint(smiles, fingerprint_type=fingerprint_type, nBits=nBits)
        if values is None:
            failed.append((cas, smiles))
            continue

        fingerprints.append([cas, smiles] + values)

    fp_len = len(fingerprints[0]) - 2 if fingerprints else (nBits if fingerprint_type != "MACCS" else 167)
    columns = ['CAS', 'Canonical SMILES'] + [f'{fingerprint_type}_{i}' for i in range(fp_len)]

    fingerprint_df = pd.DataFrame(fingerprints, columns=columns)
    fingerprint_df.to_csv(output_path, index=False)
    print(f"Saved {fingerprint_type} fingerprints to {output_path}. Failed entries: {len(failed)}")
    return fingerprint_df

df = pd.read_csv('./datasets/integrated_dataset_filled.csv')

build_rdkit_fingerprint_file(df, './datasets/chemical_fingerprints_morgan.csv', fingerprint_type="Morgan", nBits=2048)
build_rdkit_fingerprint_file(df, './datasets/chemical_fingerprints_maccs.csv', fingerprint_type="MACCS")
build_rdkit_fingerprint_file(df, './datasets/chemical_fingerprints_rdkit.csv', fingerprint_type="RDKit", nBits=1024)
