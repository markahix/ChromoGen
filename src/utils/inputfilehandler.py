from datetime import datetime 
import os
# from enum import Enum

# class ModelOpt(Enum):
#     RECURRENT = 1
#     GPT = 2
#     TRANSFORMER = 3

# class TaskOpt(Enum):
#     REGULAR = 1
#     CONSTRAINED = 2

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
        self.model                        = "GPT"
        self.use_scaffold                 = False
        self.log_level                    = "log" # "verbose", "debug"
        self.logfile_name                 = "logfile.out"
    
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
        print("Input file not found.  Generating template file at given location.")
        quit()
    
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
                exec(f"self.{key} = {val}")
        if type(self.multipliers) != list:
            self.multipliers = [self.multipliers]
        if type(self.reward_fns) != list:
            self.reward_fns = [self.reward_fns]
        if type(self.predictor_paths) != list:
            self.predictor_paths = [self.predictor_paths]
        self.ValidateSettings()

    def ValidateSettings(self):
        invalid_settings = []
        if type(self.batch_size) != int:
            invalid_settings.append("batch_size")
        if type(self.epochs) != int:
            invalid_settings.append("epochs")
        if type(self.learning_rate) != float:
            invalid_settings.append("learning_rate")
        if type(self.load_pretrained) != bool:
            invalid_settings.append("load_pretrained")
        if type(self.do_train) != bool:
            invalid_settings.append("do_train")
        if type(self.pretrained_path) != str:
            invalid_settings.append("pretrained_path")
        if self.tokenizer not in ["Char","BPE"]:
            invalid_settings.append("tokenizer")
        if type(self.rl_batch_size) != int:
            invalid_settings.append("rl_batch_size")
        if type(self.rl_epochs) != int:
            invalid_settings.append("rl_epochs")
        if type(self.discount_factor) != float:
            invalid_settings.append("discount_factor")
        if type(self.rl_max_len) != int:
            invalid_settings.append("rl_max_len")
        if type(self.rl_size) != int:
            invalid_settings.append("rl_size")
        for item in self.reward_fns:
            if item not in ['QED', 'Sim', 'Anti Cancer', 'LIDI', 'Docking', 'AbsorptionReward', 'EmissionReward', 'QuantumYieldReward']:
                invalid_settings.append(f"reward_fns - {item}")
        if type(self.do_eval) != bool:
            invalid_settings.append("do_eval")
        if type(self.eval_steps) != int:
            invalid_settings.append("eval_steps")
        if type(self.rl_temperature) != int:
            invalid_settings.append("rl_temperature")
        if type(self.multipliers) != list:
            invalid_settings.append("multipliers")
        if type(self.no_batched_rl) != bool:
            invalid_settings.append("no_batched_rl")
        if type(self.predictor_paths) != list:
            invalid_settings.append("predictor_paths")
        if type(self.save_path) != str:
            invalid_settings.append("save_path")
        if type(self.eval_size) != int:
            invalid_settings.append("eval_size")
        if type(self.eval_max_len) != int:
            invalid_settings.append("eval_max_len")
        if type(self.temperature) != int:
            invalid_settings.append("temperature")
        if type(self.n_embd) != int:
            invalid_settings.append("n_embd")
        if type(self.d_model) != int:
            invalid_settings.append("d_model")
        if type(self.n_layers) != int:
            invalid_settings.append("n_layers")
        if type(self.num_heads) != int:
            invalid_settings.append("num_heads")
        if type(self.block_size) != int:
            invalid_settings.append("block_size")
        if type(self.proj_size) != int:
            invalid_settings.append("proj_size")
        if type(self.attn_dropout_rate) != float:
            invalid_settings.append("attn_dropout_rate")
        if type(self.proj_dropout_rate) != float:
            invalid_settings.append("proj_dropout_rate")
        if type(self.resid_dropout_rate) != float:
            invalid_settings.append("resid_dropout_rate")
        if type(self.predictor_dataset_path) != str:
            invalid_settings.append("predictor_dataset_path")
        if type(self.predictor_tokenizer_path) != str:
            invalid_settings.append("predictor_tokenizer_path")
        if type(self.predictor_save_path) != str:
            invalid_settings.append("predictor_save_path")
        if type(self.train_predictor) != bool:
            invalid_settings.append("train_predictor")
        if type(self.predictor_batch_size) != int:
            invalid_settings.append("predictor_batch_size")
        if type(self.predictor_epochs) != int:
            invalid_settings.append("predictor_epochs")
        if type(self.predictor_n_embd) != int:
            invalid_settings.append("predictor_n_embd")
        if type(self.predictor_d_model) != int:
            invalid_settings.append("predictor_d_model")
        if type(self.predictor_n_layers) != int:
            invalid_settings.append("predictor_n_layers")
        if type(self.predictor_num_heads) != int:
            invalid_settings.append("predictor_num_heads")
        if type(self.predictor_block_size) != int:
            invalid_settings.append("predictor_block_size")
        if type(self.predictor_proj_size) != int:
            invalid_settings.append("predictor_proj_size")
        if type(self.predictor_attn_dropout_rate) != float:
            invalid_settings.append("predictor_attn_dropout_rate")
        if type(self.predictor_proj_dropout_rate) != float:
            invalid_settings.append("predictor_proj_dropout_rate")
        if type(self.predictor_resid_dropout_rate) != float:
            invalid_settings.append("predictor_resid_dropout_rate")
        if type(self.dataset_path) != str:
            invalid_settings.append("dataset_path")
        if type(self.tokenizer_path) != str:
            invalid_settings.append("tokenizer_path")
        if type(self.device) != str :
            invalid_settings.append("device")
        if self.model not in ["RECURRENT","GPT","TRANSFORMER"]:
            invalid_settings.append("model")
        if type(self.use_scaffold) != bool:
            invalid_settings.append("use_scaffold")
        if self.log_level not in ["log","verbose","debug"]:
            self.log_level = "debug"
            print("Log Level setting unrecognized.  Set to DEBUG.")
        if invalid_settings == []:
            print("Settings validated.")
        else:
            print("Invalid settings detected:")
            for i in invalid_settings:
                print(f"\t{i}")
            quit()
                     