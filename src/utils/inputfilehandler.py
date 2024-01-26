from datetime import datetime 
import os
from enum import Enum

class ModelOpt(Enum):
    RECURRENT = 1
    GPT = 2
    TRANSFORMER = 3

class TaskOpt(Enum):
    REGULAR = 1
    CONSTRAINED = 2

class Arguments():
    def __init__(self):
        self.batch_size                   = 512
        self.epochs                       = 3
        self.learning_rate                = 1e-3
        self.load_pretrained              = False
        self.do_train                     = False
        self.pretrained_path              = './data/models/gpt_pre_rl_gdb13.pt'
        self.tokenizer                    = "Char"
        self.rl_batch_size                = 500
        self.rl_epochs                    = 100
        self.discount_factor              = 0.99
        self.rl_max_len                   = 150
        self.rl_size                      = 25000
        self.reward_fns                   = ["AbsorptionReward","EmissionReward","QuantumYieldReward"]
        self.do_eval                      = False
        self.eval_steps                   = 10
        self.rl_temperature               = 1
        self.multipliers                  = ["lambda x: x"]
        self.no_batched_rl                = False
        self.predictor_paths              = [None]
        self.save_path                    = './data/results/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.eval_size                    = 25000
        self.eval_max_len                 = 150
        self.temperature                  = 1
        self.n_embd                       = 512
        self.d_model                      = 1024
        self.n_layers                     = 4
        self.num_heads                    = 8
        self.block_size                   = 512
        self.proj_size                    = 256
        self.attn_dropout_rate            = 0.1
        self.proj_dropout_rate            = 0.1
        self.resid_dropout_rate           = 0.1
        self.predictor_dataset_path       = './data/csvs/bs1.csv'
        self.predictor_tokenizer_path     = './data/tokenizers/predictor_tokenizer.json'
        self.predictor_save_path          = './data/models/predictor_model.pt'
        self.train_predictor              = False
        self.predictor_batch_size         = 32
        self.predictor_epochs             = 10
        self.predictor_n_embd             = 512
        self.predictor_d_model            = 1024
        self.predictor_n_layers           = 4
        self.predictor_num_heads          = 8
        self.predictor_block_size         = 512
        self.predictor_proj_size          = 256
        self.predictor_attn_dropout_rate  = 0.1
        self.predictor_proj_dropout_rate  = 0.1
        self.predictor_resid_dropout_rate = 0.1
        self.dataset_path                 = './data/gdb/gdb13/gdb13.smi'
        self.tokenizer_path               = './data/tokenizers/gdb13ScaffoldCharTokenizer.json'
        self.device                       = 'cuda'
        self.model                        = ModelOpt.GPT
        self.use_scaffold                 = False
        self.log_level                    = "default" # "verbose", "debug"
        print(type(self.model))
    def CreateTemplateFile(self,inputfile):
        with open(inputfile,"w") as f:
            f.write("##################################### \n")
            f.write("# Input file template for ChromoGen # \n")
            f.write("##################################### \n")
            for key,val in self.__dict__.items():
                if type(val) == list:
                    try:
                        tmp = ','.join([x for x in val])
                    except:
                        tmp = "None"
                    f.write(f"{key:<30} {tmp}\n")
                else:
                    f.write(f"{key:<30} {val}\n")
    def ReadInputFile(self,inputfile):
        if not os.path.exists(inputfile):
            self.CreateTemplateFile(inputfile)
        else:
            for line in open(inputfile).readlines():
                if line.strip()[0] == "#":
                    continue
                [key,val] = line.strip().split(' ',1)
                key = key.strip()
                val = val.strip()
                if key == "model":
                    exec(f"args.{key} = {val}")
                else:
                    if type(val) == str:
                        if val == "False":
                            val = False
                        elif val == "True":
                            val = True
                        else:
                            try:
                                val = int(val)
                            except:
                                try:
                                    val = float(val)
                                except:
                                    if ',' in val:
                                        val = val.split(",")
                                    else:
                                        val = "\'" + val + "\'"
                    exec(f"args.{key} = {val}")
            self.ValidateSettings()
        if type(self.multipliers) != list:
            self.multipliers = [self.multipliers]
        if type(self.reward_fns) != list:
            self.reward_fns = [self.reward_fns]
        if type(self.predictor_paths) != list:
            self.predictor_paths = [self.predictor_paths]
    def ValidateSettings(self):
        invalid_settings = []
        tests = [type(self.batch_size) == int,
                type(self.epochs) == int,
                type(self.learning_rate) == float,
                type(self.load_pretrained) == bool,
                type(self.do_train) == bool,
                type(self.pretrained_path) == str,
                type(self.tokenizer) == str,
                type(self.rl_batch_size) == int,
                type(self.rl_epochs) == int,
                type(self.discount_factor) == float,
                type(self.rl_max_len) == int,
                type(self.rl_size) == int,
                type(self.reward_fns) == list,
                type(self.do_eval) == bool,
                type(self.eval_steps) == int,
                type(self.rl_temperature) == int,
                type(self.multipliers) == list,
                type(self.no_batched_rl) == bool,
                type(self.predictor_paths) == list,
                type(self.save_path) == str,
                type(self.eval_size) == int,
                type(self.eval_max_len) == int,
                type(self.temperature) == int,
                type(self.n_embd) == int,
                type(self.d_model) == int,
                type(self.n_layers) == int,
                type(self.num_heads) == int,
                type(self.block_size) == int,
                type(self.proj_size) == int,
                type(self.attn_dropout_rate) == float,
                type(self.proj_dropout_rate) == float,
                type(self.resid_dropout_rate) == float,
                type(self.predictor_dataset_path) == str,
                type(self.predictor_tokenizer_path) == str,
                type(self.predictor_save_path) == str,
                type(self.train_predictor) == bool,
                type(self.predictor_batch_size) == int,
                type(self.predictor_epochs) == int,
                type(self.predictor_n_embd) == int,
                type(self.predictor_d_model) == int,
                type(self.predictor_n_layers) == int,
                type(self.predictor_num_heads) == int,
                type(self.predictor_block_size) == int,
                type(self.predictor_proj_size) == int,
                type(self.predictor_attn_dropout_rate) == float,
                type(self.predictor_proj_dropout_rate) == float,
                type(self.predictor_resid_dropout_rate) == float,
                type(self.dataset_path) == str,
                type(self.tokenizer_path) == str,
                type(self.device) == str ,
                type(self.model) == ModelOpt,
                type(self.use_scaffold) == bool]
        if all(tests):
            print("Settings validated.")
        else:
            print("Invalid settings detected.")
                     