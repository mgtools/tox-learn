import numpy as np
import re
from decimal import *
import requests

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

def conformation_array(smiles, conf_type='etkdg'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return False, None, None

    mol_from_smiles = Chem.AddHs(mol)
    if mol_from_smiles is None:
        print(f"Failed to add hydrogens for SMILES: {smiles}")
        return False, None, None

    if conf_type == 'etkdg':
        status = AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDG())
    elif conf_type == 'etkdgv3':
        status = AllChem.EmbedMolecule(mol_from_smiles, AllChem.ETKDGv3())
    elif conf_type == 'omega':
        raise ValueError('OMEGA conformation will be supported soon.')
    else:
        raise ValueError('Unsupported conformation type: {}'.format(conf_type))

    if status == -1:
        print(f"Failed to generate conformer for SMILES: {smiles}")
        return False, None, None

    try:
        conf = mol_from_smiles.GetConformer()
    except:
        print(f"Failed to get conformer or coordinates for SMILES: {smiles}")
        return False, None, None

    xyz_arr = conf.GetPositions()
    centroid = np.mean(xyz_arr, axis=0)
    xyz_arr -= centroid

    xyz_arr = xyz_arr.tolist()
    
    for i, atom in enumerate(mol_from_smiles.GetAtoms()):
        xyz_arr[i] += [atom.GetDegree(), atom.GetExplicitValence(), atom.GetMass() / 100,
                       atom.GetFormalCharge(), atom.GetNumImplicitHs(),
                       int(atom.GetIsAromatic()), int(atom.IsInRing())] 
    xyz_arr = np.array(xyz_arr)

    atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
    return True, xyz_arr, atom_type
def smiles_to_iupac(smiles):
	global CACTUS 
	rep = "iupac_name"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""

def smiles_to_inchi(smiles):
	global CACTUS 
	rep = "stdinchi"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""

def smiles_to_inchikey(smiles):
	global CACTUS 
	rep = "stdinchikey"
	url = CACTUS.format(smiles, rep)
	try: 
		response = requests.get(url)
		response.raise_for_status()
		return response.text
	except:
		print("Sorry, your structure identifier could not be resolved (the request returned a HTML 404 status message)")
		return ""

def conformation_arrays(smiles, num_confs=3, conf_type='etkdg'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return False, None, None

    mol_with_h = Chem.AddHs(mol)
    if mol_with_h is None:
        print(f"Failed to add hydrogens for SMILES: {smiles}")
        return False, None, None

    embedding_options = None
    if conf_type == 'etkdg':
        embedding_options = AllChem.ETKDG()
        embedding_options.numThreads = 4
    elif conf_type == 'etkdgv3':
        embedding_options = AllChem.ETKDGv3()
    elif conf_type == 'omega':
        raise ValueError('OMEGA conformation will be supported soon.')
    else:
        raise ValueError('Unsupported conformation type: {}'.format(conf_type))

    status = AllChem.EmbedMultipleConfs(mol_with_h, numConfs=num_confs, params=embedding_options)
    if len(status) == 0:
        print(f"Failed to generate conformer for SMILES: {smiles}")
        return False, None, None

    conformations = []
    for conf_id in status:
        try:
            conf = mol_with_h.GetConformer(conf_id)
            xyz_arr = conf.GetPositions()
            centroid = np.mean(xyz_arr, axis=0)
            xyz_arr -= centroid
            xyz_arr = xyz_arr.tolist()

            for i, atom in enumerate(mol_with_h.GetAtoms()):
                xyz_arr[i] += [atom.GetDegree(), atom.GetExplicitValence(), atom.GetMass() / 100,
                               atom.GetFormalCharge(), atom.GetNumImplicitHs(),
                               int(atom.GetIsAromatic()), int(atom.IsInRing())] 

            xyz_arr = np.array(xyz_arr)
            conformations.append(xyz_arr)
        except:
            print(f"Failed to get conformer or coordinates for SMILES: {smiles}")
            return False, None, None

    atom_type = [atom.GetSymbol() for atom in mol_with_h.GetAtoms()]
    return True, conformations, atom_type
# Test the conformation_array function
def test_conformation_array():
    smiles_list = [
        "CCN(CC)C(=O)SCC1=CC=C(Cl)C=C1",
        # "CNC(=O)OC1=C2C=CC=CC2=CC=C1",
        # "CCO",  
        # "invalid_smiles"  
    ]
    
    for smiles in smiles_list:
        success, coords, atom_types = conformation_arrays(smiles)
        if success:
            print(f"Conformer generated for SMILES: {smiles}")
            print("Coordinates:\n", coords)
            print("Atom Types:", atom_types)
        else:
            print(f"Failed to generate conformer for SMILES: {smiles}")

# if __name__ == "__main__":
#     test_conformation_array()
	# smiles = "CCO"
	# mol = Chem.MolFromSmiles(smiles)
	# mol = Chem.AddHs(mol)

	# # Using ETKDG method for conformer generation
	# status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

	# if status != -1:
	# 	print("Conformer generation successful!")
	# 	conf = mol.GetConformer()
	# 	for i in range(mol.GetNumAtoms()):
	# 		pos = conf.GetAtomPosition(i)
	# 		print(f"Atom {i}: x={pos.x}, y={pos.y}, z={pos.z}")
	# else:
	# 	print("Conformer generation failed!")