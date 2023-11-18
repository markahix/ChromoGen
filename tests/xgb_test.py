import xgboost as xgb
import numpy as np
from rdkit import Chem
from map4 import MAP4Calculator

# Function to convert SMILES to MAP4 fingerprint
def get_map4(smiles_string, dimensions=1024):
    mol = Chem.MolFromSmiles(smiles_string)
    map4_calculator = MAP4Calculator(dimensions=dimensions)
    if mol:
        map4 = map4_calculator.calculate(mol)
        return map4
    return None

# Load model
def load_xgb_model(filepath):
    model = xgb.Booster()
    model.load_model(filepath)
    feature_names = model.feature_names
    return model, feature_names

# Example SMILES string
smiles_string = "CCO"  # Replace with a test molecule

# Convert SMILES to MAP4 fingerprint
map4_fp = get_map4(smiles_string)
if map4_fp is not None:
    map4_array = np.array(map4_fp).reshape(1, -1)
    dmatrix_map4_array = xgb.DMatrix(map4_array)

# Load models and get feature names
emission_model, emi_features = load_xgb_model('./data/models/emission_model.json')
absorption_model, abs_features = load_xgb_model('./data/models/absorption_model.json')
quantum_yield_model, qy_features = load_xgb_model('./data/models/quantum_yield_model.json')

# Ensure the map4_array has the correct shape and feature names
if map4_fp is not None:
    map4_array = np.array(map4_fp).reshape(1, -1)
    dmatrix_map4_array_emission = xgb.DMatrix(map4_array, feature_names=emi_features)
    dmatrix_map4_array_absorption = xgb.DMatrix(map4_array, feature_names=abs_features)
    dmatrix_map4_array_quantum_yield = xgb.DMatrix(map4_array, feature_names=qy_features)

    # Predict properties
    emission_pred = emission_model.predict(dmatrix_map4_array_emission)
    absorption_pred = absorption_model.predict(dmatrix_map4_array_absorption)
    quantum_yield_pred = quantum_yield_model.predict(dmatrix_map4_array_quantum_yield)

    # Print predictions
    print("Emission:", emission_pred)
    print("Absorption:", absorption_pred)
    print("Quantum Yield:", quantum_yield_pred)
else:
    print("Invalid SMILES or MAP4 fingerprint")
