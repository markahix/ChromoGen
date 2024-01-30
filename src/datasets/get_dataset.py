from .smiles_dataset import SmilesDataset
# from ..utils.utils import ModelOpt, TaskOpt

def get_dataset(**kwargs): # type=ModelOpt.GPT, task=TaskOpt.CONSTRAINED ### removed arguments

	dataset = SmilesDataset(**kwargs)

	return dataset
