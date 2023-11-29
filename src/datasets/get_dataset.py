from .smiles_dataset import SmilesDataset
from ..utils.utils import ModelOpt, TaskOpt

def get_dataset(type=ModelOpt.GPT, task=TaskOpt.CONSTRAINED, **kwargs):

	dataset = SmilesDataset(**kwargs)

	return dataset
