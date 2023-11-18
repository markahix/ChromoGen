from rdkit import Chem
from ..src.train.evaluate import calc_num_rings, calc_set_stat

# Test molecules in SMILES format
test_smiles = [
    "C1=CC=CC=C1",  # Benzene
    "C1CCCCC1",     # Cyclohexane
    "O1CCOCC1",     # 1,4-Dioxane
    "C1=CC=C(C=C1)C2=CC=CC=C2",  # Biphenyl
    "C1CC2CCC1C2",  # Decalin
    "C(C(=O)O)N",   # Acetamide (no rings)
    "C1NCC2=CC=CC=C2N1",  # Quinoline
]

def test_calc_set_stat(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    # Filter out None values which represent invalid SMILES
    valid_mols = [mol for mol in mols if mol is not None]
    
    # Apply calc_set_stat with calc_num_rings
    _, stats = calc_set_stat(valid_mols, calc_num_rings, lst=False, desc='Ring Count Test')
    
    return stats

# Test the calc_set_stat function
stats = test_calc_set_stat(test_smiles)
print(stats)
