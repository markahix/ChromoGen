from .recurrent import RecurrentConfig, RecurrentModel
from .gpt import GPTConfig, GPT
from .transformer import TransformerConfig, Transoformer
# from ..utils.utils import ModelOpt

def get_model(modeltype="RECURRENT", **kwargs):
	if modeltype == "GPT":
		config = GPTConfig(**kwargs)
		model = GPT(config)
	elif modeltype == "TRANSFORMER":
		config = TransformerConfig(**kwargs)
		model = Transoformer(config)
	else:
		config = RecurrentConfig(**kwargs)
		model = RecurrentModel(config)
	return model
	
