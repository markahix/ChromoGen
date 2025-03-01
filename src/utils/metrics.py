import os
import sys
from typing import List, Union

from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdchem
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def calc_num_rings(mol: Chem.rdchem.Mol) -> int:
    """
    Calculates the number of rings in a molecule. This is useful as chromophores usually have many rings.
    """
    return rdchem.Mol.GetRingInfo(mol).NumRings() if mol else 0

def calc_novelty(train_set: Union[str, List[str]], generated_molecules: List[str]) -> float:
    """
    Calculates the novelty score of the generated molecule list.
    the novelty score is the ratio between the amount of molecules that aren't in the train set
    and the size of the entire generated molecule list.

    |gen_molecules - train_set_molecules|
    -------------------------------------
               |gen_molecules|

    Args:
        train_path:
            The data used to train the model with or a list of the loaded molecules

        generated_molecules:
            List of molecules that were generated by the model, molecules are in SMILES form.

    Returns:
        The novelty score of the generated set.
        novelty score ranges between 0 and 1.
    """ 
    return len(set(generated_molecules) - set(train_set)) / len(generated_molecules)

def calc_diversity(gen_molecules: List[str]) -> float:
    """
    Calculates the diversity of the generate molecule list.
    The diversity is the number of unique molecules in the generated list.

    |unique(gen_molecules)|
    -----------------------
        |gen_molecules|

    Args:
        gen_molecules:
            List of molecules that werte generated by the model, molecules are in SMILES form.

    Returns:
        The diversity score of the generated set.
        diversity score ranges between 0 and 1.
    """ 
    return len(set(gen_molecules)) / len(gen_molecules)

# def calc_logp(mol: Chem.rdchem.Mol) -> float: ### This function is never used in the current build.  Flagging for deletion.
#     """
#     Calculates the logP for a given molecule.

#     Args:
#         mol:
#             An rdkit molecule object.

#     Returns:
#         the molecule log p which is his the log ratio between
#         water solubility and octanol solubility.
#     """
#     return Crippen.MolLogP(mol)

# def calc_qed(mol: Chem.rdchem.Mol) -> float: ### Since this function is just returning another function with no additional work, i've replaced all calls to it with the original function instead.  updated those imports as well.
#     """
#     Calculates the quantitative estimation of drug-likeness of a given molecule.

#     Args:
#         mol:
#             An rdkit molecule object.

#     Returns:
#         the molecule qed value which estimate how much this molecule
#         resemebles a drug.
#     """
#     return QED.qed(mol)

def calc_sas(mol: Chem.rdchem.Mol) -> float:
    """
    Calculates the Synthetic Accessiblity Score (SAS) of a drug-like molecule
    based on the molecular compelxity and fragment contribution.
    
    code taken from: https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py

    Args:
        A rdkit molecule object.

    Returns:
        The SAS score of a given molecule.

    Raises

    """
    try:
        sascore = sascorer.calculateScore(mol)
        return sascore
    except Exception:
        return -1

def calc_valid_molecules(molecules: List[str]) -> float:
    valid_molecules = [mol for mol in molecules if Chem.MolFromSmiles(mol) is not None]

    return len(valid_molecules) / len(molecules)
