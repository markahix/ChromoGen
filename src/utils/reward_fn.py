from abc import ABC, abstractmethod
from collections import OrderedDict

from typing import Callable, Optional, List

import chemprop

from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm.auto import tqdm
from vina import Vina

import numpy as np
import xgboost as xgb
from map4 import MAP4Calculator

from itertools import zip_longest



# Define MAP4 Function
def get_map4(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    map4_calculator = MAP4Calculator(dimensions=1024)
    if mol:
        map4 = map4_calculator.calculate(mol)
        return map4
    return None
    
def load_xgb_model(filepath):
    try:
        model = xgb.Booster()
        model.load_model(filepath)
        print(f"Model loaded successfully from {filepath}")
        # Get feature names from the model
        feature_names = model.feature_names
        return model, feature_names
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None, None

def get_reward_fn(reward_names: List[str], paths: List[str]=None, multipliers: List[str]=None, **kwargs):
    reward_fns = []
    for reward_name, path, mult in zip_longest(reward_names, paths, multipliers, fillvalue=None):
        if reward_name == 'Anti Cancer':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))

        elif reward_name == 'LIDI':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))

        elif reward_name == 'Sim':
            reward_fn = SimilarityReward(path, reward_name, multiplier=eval(mult))
        
        elif reward_name == 'QED':
            reward_fn = QEDReward(reward_name, multiplier=eval(mult))

        elif reward_name == 'Docking':
            reward_fn = DockingReward(path, reward_name, multiplier=eval(mult))
            
        elif reward_name == 'EmissionReward':
            evaluated_multiplier = eval(mult) if mult is not None else None
            reward_fn = EmissionReward(multiplier=evaluated_multiplier)
            
        elif reward_name == 'AbsorptionReward':
            evaluated_multiplier = eval(mult) if mult is not None else None
            reward_fn = AbsorptionReward(multiplier=evaluated_multiplier)
            
        elif reward_name == 'QuantumYieldReward':
            evaluated_multiplier = eval(mult) if mult is not None else None
            reward_fn = QuantumYieldReward(multiplier=evaluated_multiplier)
        
        reward_fns.append(reward_fn)

    if len(reward_fns) == 1:
        print("Single reward function chosen")
        return reward_fns[0]
    else:
        print("Multipe Reward functions chosen")
        return MultiReward(name='MultiReward', reward_fns=reward_fns)

class Reward(ABC):
    def __init__(self, name, multiplier:  Optional[Callable[[float], float]]=None, eval_: bool=False, **kwargs) -> None:
        self.name = name
        self.multiplier = multiplier
        self._eval = eval_
        
    @abstractmethod
    def __call__(self, smiles: str):
        raise NotImplementedError

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, val):
        self._eval = val

    def __str__(self,):
        return self.name

class EmissionReward(Reward):
    def __init__(self, desired_emission=1000, weight=1, multiplier=None):
        super().__init__(name="EmissionReward", multiplier=multiplier)
        self.desired_emission = desired_emission
        self.weight = weight
        self.model, self.features = load_xgb_model('./data/models/emission_model.json')

    def __call__(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        rewards = []
        for smiles in smiles_list:
            map4_fp = get_map4(smiles)
            if map4_fp is None:
                rewards.append(-10000.0)
                continue

            dmatrix = xgb.DMatrix(np.array(map4_fp).reshape(1, -1), feature_names=self.features)
            predicted_emission = self.model.predict(dmatrix)[0]
            reward = -self.weight * abs(self.desired_emission - predicted_emission)
            rewards.append(reward)

        return rewards

class AbsorptionReward(Reward):
    def __init__(self, desired_absorption=800, weight=1, multiplier=None):
        super().__init__(name="AbsorptionReward", multiplier=multiplier)
        self.desired_absorption = desired_absorption
        self.weight = weight
        self.model, self.features = load_xgb_model('./data/models/absorption_model.json')

    def __call__(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        rewards = []
        for smiles in smiles_list:
            map4_fp = get_map4(smiles)
            if map4_fp is None:
                rewards.append(-10000.0)
                continue

            dmatrix = xgb.DMatrix(np.array(map4_fp).reshape(1, -1), feature_names=self.features)
            predicted_absorption = self.model.predict(dmatrix)[0]
            reward = -self.weight * abs(self.desired_absorption - predicted_absorption)
            rewards.append(reward)

        return rewards

class QuantumYieldReward(Reward):
    def __init__(self, desired_quantum_yield=0.99, weight=500, multiplier=None):
        super().__init__(name="QuantumYieldReward", multiplier=multiplier)
        self.desired_quantum_yield = desired_quantum_yield
        self.weight = weight
        self.model, self.features = load_xgb_model('./data/models/quantum_yield_model.json')

    def __call__(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        rewards = []
        for smiles in smiles_list:
            map4_fp = get_map4(smiles)
            if map4_fp is None:
                rewards.append(-10000.0)
                continue

            dmatrix = xgb.DMatrix(np.array(map4_fp).reshape(1, -1), feature_names=self.features)
            predicted_quantum_yield = self.model.predict(dmatrix)[0]
            reward = -self.weight * abs(self.desired_quantum_yield - predicted_quantum_yield)
            rewards.append(reward)

        return rewards

class MultiReward(Reward):
    def __init__(self, name, reward_fns) -> None:
        super().__init__(name=name)
        self.reward_fns = reward_fns

    def __call__(self, smiles):
        rewards = OrderedDict()
        for fn in self.reward_fns:
            rewards[str(fn)] =  fn(smiles)

        if not self.eval:
            rewards = list(zip(*list(rewards.values())))
            rewards = [sum(rewards) for rewards in rewards]

        return rewards

    @Reward.eval.setter
    def eval(self, val):
        for fn in self.reward_fns:
            if hasattr(fn, '_eval'):
                fn.eval = val
        Reward.eval.fset(self, val)

class DockingReward(Reward):
    def __init__(self, receptor_path, name, multiplier=None, **kwargs):
        super().__init__(name=f"{receptor_path.split('/')[-1].split('.')[0]}_{name}", multiplier=multiplier, **kwargs)

        self.receptor_path = receptor_path
        protein = Chem.MolFromPDBFile(receptor_path[:-2]) #we remove the last 2 chars with will resutls in reading the PDB file
        pos = protein.GetConformer(0).GetPositions()
        self.center = (pos.max(0) + pos.min(0)) / 2
        print(f"Protein center is: {self.center}")

        self.vina = Vina(sf_name='vina', cpu=0, verbosity=1)

    def __call__(self, smiles: List[str]):
        if isinstance(smiles, str):
            print("Converting smiles to list")
            smiles = [smiles]

        rewards = [self.__dock(s) if Chem.MolFromSmiles(s) is not None else 0 for s in tqdm(smiles)]

        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

    def __dock(self, smiles):
        try:
            # Create RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol)

            else:
                return 0

            # Prepare mol
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            for setup in mol_setups:
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)

            if not is_ok:
                return 0
        
            # Configure Vina
            self.vina.set_receptor(self.receptor_path)
            self.vina.set_ligand_from_string(pdbqt_string)

            # Define the search space (coordinates and dimensions)
            x, y, z = self.center
            self.vina.compute_vina_maps(center=[x, y, z], box_size=[30, 30, 30])

            # Run docking
            self.vina.dock(n_poses=5, exhaustiveness=32)

            return self.vina.score()[0]

        except Exception:
            return 0


class SimilarityReward(Reward):
    def __init__(self, smiles, name, multiplier=None, **kwargs):
        super().__init__(name=name, multiplier=multiplier, **kwargs)

        self.name = name
        self.smiles = smiles
        self.fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))

    def __call__(self, smiles: List[str]):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        filtered_mols = [mol for mol in mols if mol is not None]
        fps = [Chem.RDKFingerprint(mol) for mol in filtered_mols]
        sims = DataStructs.BulkTanimotoSimilarity(self.fp, fps)

        rewards = [sims[filtered_mols.index(mols[i])] if mols[i] is not None else 0 for i in range(len(mols))]

        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

class ChempropReward(Reward):
    def __init__(self,
                predictor_path,
                name: str='Chemprop',
                multiplier: Optional[Callable[[float], float]]=None,
                **kwargs) -> None:
        super().__init__(name=name, multiplier=multiplier, **kwargs)
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', predictor_path,
            #'--features_generator', 'rdkit_2d_normalized',
            '--no_features_scaling',
        ]

        self.args = chemprop.args.PredictArgs().parse_args(arguments)

        self.model_objects = chemprop.train.load_model(args=self.args)

    def __call__(self, smiles: str) -> float:
        if isinstance(smiles, list):
            smiles = [[s] for s in smiles]
        else:
            smiles = [[smiles]]
        preds = []
        try:
            preds = chemprop.train.make_predictions(args=self.args, smiles=smiles, model_objects=self.model_objects)
            preds = [pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0 for pred in preds]
        except TypeError:
            print('Faild to make predictions')
            for s in smiles:
                try:
                    pred = chemprop.train.make_predictions(args=self.args, smiles=[s], model_objects=self.model_objects)
                    preds.append(pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0)
                except TypeError:
                    print(f'Bad SMILES: {s[0]}')
                    preds.append(0)

        if self.multiplier is not None and not self.eval:
            #print(preds[:5])
            preds = [self.multiplier(pred) for pred in preds]
            #print(preds[:5])

        return preds

class QEDReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):

        super(QEDReward, self).__init__(name=name, multiplier=multiplier)

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [qed(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

## Is anything below this line getting used, or can we delete it entirely?
def main():
    with QEDReward() as rfn:
        print(rfn("CCO"))

    with QEDReward() as rfn:
        print(rfn("CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]=]"))

        
if __name__ == "__main__":
    main()