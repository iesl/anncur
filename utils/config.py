import os
import json
import torch
import random
import argparse
import warnings
import numpy as np



class BaseConfig(object):
	
	def __init__(self, filename=None):
		
		self.config_name = filename
		self.seed 			= 1234
		
		
	def to_json(self):
		return json.dumps(filter_json(self.__dict__), indent=4, sort_keys=True)

	def save_config(self, res_dir, filename='config.json'):
		fname = os.path.join(res_dir, filename)
		with open(fname, 'w') as fout:
			json.dump(filter_json(self.__dict__),fout, indent=4, sort_keys=True)
		return fname
	
	def __getstate__(self):
		state = dict(self.__dict__)
		if "logger" in state:
			del state['logger']
			
		return state
	
	def update_random_seeds(self, seed):
		raise NotImplementedError
	
	@staticmethod
	def get_parser_for_args():
		
		parser = argparse.ArgumentParser(description='Get config from str')
		
		dummy_config = Config()
		################################## OPTIONAL ARGUMENTS TO OVERWRITE CONFIG FILE ARGS#############################
		for config_arg in dummy_config.__dict__:
			def_val = dummy_config.__getattribute__(config_arg)
			if type(def_val) == tuple:
				def_val = def_val[0]
				
			arg_type = type(def_val) if def_val is not None else str
			arg_type = arg_type if arg_type is not dict else str
			
			if arg_type == list or arg_type == tuple:
				arg_type = type(def_val[0]) if len(def_val) > 0 else str
				parser.add_argument('--{}'.format(config_arg), nargs='+', type=arg_type, default=None,
									help='If not specified then value from config file will be used')
			else:
				parser.add_argument('--{}'.format(config_arg), type=arg_type, default=None,
									help='If not specified then value from config file will be used')
		################################################################################################################
	
		return parser
	
	def update_config_from_arg_list(self, arg_list):
		
		parser = BaseConfig.get_parser_for_args()
		print(f"Parsing arg_list = {arg_list}\n\n")
		args = parser.parse_args(arg_list)
		
		for config_arg in self.__dict__:
			def_val = getattr(args, config_arg)
			if def_val is not None:
				
				old_val = self.__dict__[config_arg]
				self.__dict__.update({config_arg: def_val})
				new_val = self.__dict__[config_arg]
				print("Updating Config.{} from {} to {} using arg_val={}".format(config_arg, old_val, new_val, def_val))
		
		self.update_random_seeds(self.seed)

	
class Config(BaseConfig):
	def __init__(self, filename=None):
		
		super(Config, self).__init__(filename=filename)
		self.config_name 	= filename
		
		self.save_code		= True
		self.base_res_dir	= "../../results"
		self.exp_id			= ""
		self.res_dir_prefix	= "" # Prefix to add to result dir name
		self.misc			= ""
		
		self.seed 			= 1234
		self.n_procs		= 20
		
		self.max_time = "06:23:55:00" # 7 days - 5 minutes
		self.fast_dev_run = 0 # Run a few batches from train/dev for sanity check
		
		self.print_interval = 10
		self.eval_interval  = 800.0
		
		
		# Data specific params
		self.data_type 		= "dummy"
		self.data_dir		= "None"
		self.trn_files   = {"dummy_domain":("dummy_ment_file", "dummy_ent_file", "dummy_ent_tokens_file")}
		self.dev_files   = {"dummy_domain":("dummy_ment_file", "dummy_ent_file", "dummy_ent_tokens_file")}
		
		self.train_domains = ["dummy"],
		self.dev_domains = ["dummy"],
		self.mention_file_template = "",
		self.entity_file_template = ""
		self.entity_token_file_template = "",
		
		
		self.mode = "train"
		self.debug_w_small_data = 0
		
		# Model/Optimization specific params
		self.use_GPU  		= True
		self.num_gpus		= 1
		self.strategy		= ""
		self.type_optimization = ""
		self.learning_rate = 0.00001
		self.weight_decay = 0.01
		self.fp16 = False
		
		self.ckpt_path = ""
		self.model_type	= "" # Choose between bi-encoder, cross-encoder
		self.cross_enc_type = "default"
		self.bi_enc_type = "separate" # Use "separate" encoder for query/input/mention and label/entity or "shared" encoder
		self.bert_model = "" # Choose type of bert model - bert-uncased-small etc
		self.bert_args = {} # Some arguments to pass to bert-model when initializing
		self.lowercase = True # Use lowercase BERT tokenizer
		self.shuffle_data = True # Shuffle data during training
		self.path_to_model = ""
		self.encoder_wrapper_config = ""
		
		self.num_epochs = 4
		self.warmup_proportion = 0.01
		self.train_batch_size = 16
		self.grad_acc_steps = 4
		self.max_grad_norm = 1.
		self.loss_type = "ce"
		self.hinge_margin = 0.5
		self.reload_dataloaders_every_n_epochs = 0
		self.ckpt_metric = "loss"
		self.num_top_k_ckpts = 2
		
		self.neg_strategy = "dummy" # Strategy for choosing negatives per input for a cross-/bi-encoder model
		self.num_negs = 63 # Number of negatives per input when using a cross-/bi-encoder model
		self.neg_mine_bienc_model_file = ""
		
	
		# Parameters for distillation
		self.ent_w_score_file_template = "" # Template name for file w/ info about labels and their scores to use for distillation. Need to fill in domain name using string format option for this to work
		self.train_ent_w_score_file_template = "" # Useful when train and dev files have to be different name formats
		self.dev_ent_w_score_file_template = "" # Useful when train and dev files have to be different name format
		self.distill_n_labels = 64 # Number of labels per example to use in distillation training
		
		## BERT model specific params
		self.embed_dim = 768
		self.pooling_type = "" # Pooling on top of encoder layer to obtain input/label embedding
		self.add_linear_layer = False
		self.max_input_len = 128
		self.max_label_len = 128
		
		# Eval specific
		self.eval_batch_size = 64
		
		if filename is not None:
			with open(filename) as fin:
				param_dict = json.load(fin)
				
			self.__dict__.update({key:val for key,val in param_dict.items() if key in self.__dict__})
			extra_params = {key:val for key,val in param_dict.items() if key not in self.__dict__}
			if len(extra_params) > 0:
				warnings.warn(f"\n\nExtra params in config dict {extra_params}\n\n")
		
		self.torch_seed 	= None
		self.np_seed 		= None
		self.cuda_seed 		= None
		self.update_random_seeds(self.seed)
		
	@classmethod
	def load_from_dict(cls, param_dict):
		temp_config = cls()
		temp_config.__dict__.update(param_dict)
	
	def update_from_dict(self, param_dict):
		self.__dict__.update(param_dict)
		
	@property
	def cuda(self):
		return self.use_GPU and torch.cuda.is_available()
	
	@property
	def device(self):
		return torch.device("cuda" if self.cuda else "cpu")
	
	@property
	def result_dir(self):
		
		result_dir = "{base}/d={d}/{prefix}m={m}_l={l}_neg={neg}_s={s}{misc}".format(
			base=self.base_res_dir + "/" + self.exp_id if self.exp_id != ""
													   else self.base_res_dir,
			prefix=self.res_dir_prefix,
			d=self.data_type,
			m=self.model_type,
			l=self.loss_type,
			neg=self.neg_strategy,
			s=self.seed,
			misc="_{}".format(self.misc) if self.misc != "" else "")
		
		return result_dir
		
	@property
	def model_dir(self):
		return os.path.join(self.result_dir, "model")
	
	def update_random_seeds(self, random_seed):
	
		self.seed = random_seed
		random.seed(random_seed)
		
		self.torch_seed  = random.randint(0, 1000)
		self.np_seed     = random.randint(0, 1000)
		self.cuda_seed   = random.randint(0, 1000)
		
		torch.manual_seed(self.torch_seed)
		np.random.seed(self.np_seed)
		if self.use_GPU and torch.cuda.is_available():
			torch.cuda.manual_seed(self.cuda_seed)

	
def filter_json(the_dict):
	res = {}
	for k in the_dict.keys():
		if type(the_dict[k]) is str or \
				type(the_dict[k]) is float or \
				type(the_dict[k]) is int or \
				type(the_dict[k]) is list or \
				type(the_dict[k]) is bool or \
				the_dict[k] is None:
			res[k] = the_dict[k]
		elif type(the_dict[k]) is dict:
			res[k] = filter_json(the_dict[k])
	return res
