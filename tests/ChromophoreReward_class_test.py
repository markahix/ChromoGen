import xgboost as xgb
import numpy as np
from rdkit import Chem
from map4 import MAP4Calculator
from ..src.utils.reward_fn import ChromophoreReward

# Function to convert SMILES to MAP4 fingerprint
def get_map4(smiles_string, dimensions=1024):
    mol = Chem.MolFromSmiles(smiles_string)
    map4_calculator = MAP4Calculator(dimensions=dimensions)
    if mol:
        map4 = map4_calculator.calculate(mol)
        return map4
    return None

def test_chromophore_reward():
    # Initialize ChromophoreReward
    reward = ChromophoreReward()

    # Test SMILES strings
    test_smiles = ["CCO", "C1=CC=CC=C1", "C1=CC=C(C=C1)O"]  # Add more if needed

    # Test Model Predictions
    for smiles in test_smiles:
        map4_fp = get_map4(smiles)
        if map4_fp:
            map4_array = np.array(map4_fp).reshape(1, -1)
            print(f"Testing predictions for SMILES: {smiles}")
            # Test each model prediction
            # Add code to test emission, absorption, and quantum yield predictions

    # Compute Rewards
    for smiles in test_smiles:
        print(f"Computing reward for SMILES: {smiles}")
        reward_value = reward(smiles)
        print(f"Reward: {reward_value}")

if __name__ == "__main__":
    test_chromophore_reward()
