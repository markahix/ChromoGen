from ..src.utils.metrics import calc_num_rings
from rdkit import Chem

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

def test_calc_num_rings(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid SMILES: {smiles}"
        return calc_num_rings(mol)
    except Exception as e:
        return f"Error with {smiles}: {e}"

# Test the function
for smiles in test_smiles:
    result = test_calc_num_rings(smiles)
    print(f"SMILES: {smiles}, Number of Rings: {result}")
