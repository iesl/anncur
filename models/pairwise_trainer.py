import os
import sys
import wandb
import torch
import pprint
import logging
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import WandbLogger


from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.config import Config
from utils.data_process import load_raw_data, get_dataloader


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def load_pairwise_model(config):
	
	if config.model_type == "bi_enc":
		pairwise_model = BiEncoderWrapper(config)
		return pairwise_model
	elif config.model_type == "cross_enc":
		pairwise_model = CrossEncoderWrapper(config)
		return pairwise_model
	else:
		raise NotImplementedError(f"Support for model type = {config.model_type} not implemented")


class EntLinkData(LightningDataModule):
	
	def __init__(self, config):
		super(EntLinkData, self).__init__()
		self.config = config
		self.raw_dev_data = {}
		self.raw_train_data = {}
		
	# def prepare_data(self):
	# 	# Do not assign state in this function i.e. do not do self.x = y
	# 	pass
	
	def setup(self, stage=None):
		LOGGER.info("Inside setup function in DataModule")
		
		self.raw_train_data	= load_raw_data(config=self.config, data_split_type="train")
		self.raw_dev_data	= load_raw_data(config=self.config, data_split_type="dev")
		
		if self.config.debug_w_small_data:
			self.raw_train_data = {domain:(ments[:100], ents) for domain, (ments, ents) in self.raw_train_data.items()}
			self.raw_dev_data 	= {domain:(ments[:100], ents) for domain, (ments, ents) in self.raw_dev_data.items()}
		
		LOGGER.info("Finished setup function in DataModule")
	
	def val_dataloader(self):

		LOGGER.info("\t\tIn val_dataloader")
		biencoder = self.get_bienc_model()
		LOGGER.info("\t\tFinished biencoder and crossencoder models as needed")
		

		bienc_in_train_mode = biencoder.training if biencoder else False
		if biencoder: biencoder.eval()
		
		LOGGER.info(f"\n\n\t\tLoading validation data in DataModule\n\n")
		dev_dataloader = get_dataloader(
			raw_data=self.raw_dev_data,
			config=self.config,
			batch_size=self.config.eval_batch_size,
			shuffle_data=False,
			biencoder=biencoder,
		)
		if biencoder and bienc_in_train_mode: biencoder.train()
		
		torch.cuda.empty_cache()
		LOGGER.info("Finished loading validation data")
		return dev_dataloader
	
	def train_dataloader(self):
		
		LOGGER.info("\t\tIn train_dataloader")
		biencoder = self.get_bienc_model()
		
		# Load data in pytorch data loader.
		# Since we will take a gradient step after self.config.grad_acc_steps,
		# effective_batch_size given to data_loader = int(config.train_batch_size/self.config.grad_acc_steps)
		
		bienc_in_train_mode = biencoder.training if biencoder else False
		if biencoder: biencoder.eval()
		
		LOGGER.info(f"\n\n\t\tLoading training data in DataModule\n\n")
		train_dataloader = get_dataloader(
			config=self.config,
			raw_data=self.raw_train_data,
			batch_size=int(self.config.train_batch_size/self.config.grad_acc_steps),
			shuffle_data=self.config.shuffle_data,
			biencoder=biencoder,
		)
		if biencoder and bienc_in_train_mode: biencoder.train()
		
		
		torch.cuda.empty_cache()
		LOGGER.info("Finished loading training data")
		return train_dataloader
	
	@property
	def train_data_len(self):
		"""
		Returns number of batches in training data.
		This assumes that a batch of size b will contain positive and negative entities for b unique mentions,
		and that each mention contributes exactly one training datapoint to train_dataloader
		:return:
		"""
		if self.config.data_type in ["ent_link", "ent_link_ce"]:
			batch_size = int(self.config.train_batch_size/self.config.grad_acc_steps)
			total_ments = np.sum([len(ment_data) for domain, (ment_data, ent_data) in self.raw_train_data.items()])
			num_batches = int(total_ments/batch_size)
			return num_batches
		else:
			raise NotImplementedError(f"Data type = {self.config.data_type} not supported")
	
	def get_bienc_model(self):
		"""
		Get biencoder model to pass to data loader functions. This will be used for mining (hard) negatives for training the model
		:return:
		"""
		if self.config.model_type == "cross_enc": # Load biencoder model when training a cross-encoder model
			load_bienc_model = self.config.neg_strategy in ["bienc_hard_negs"]
		
			if load_bienc_model and os.path.isfile(self.config.neg_mine_bienc_model_file):
				LOGGER.info(f"Loading biencoder model from {self.config.neg_mine_bienc_model_file}")
				neg_mine_bienc_model = load_pairwise_model(Config(self.config.neg_mine_bienc_model_file)) \
					if self.config.neg_mine_bienc_model_file.endswith("json") \
					else BiEncoderWrapper.load_from_checkpoint(self.config.neg_mine_bienc_model_file)
				
				LOGGER.info(f"Finished loading biencoder model from {self.config.neg_mine_bienc_model_file}")
				return neg_mine_bienc_model
			else:
				LOGGER.info(f"Biencoder model will not be loaded as load_bienc_model = {load_bienc_model} and "
							f"neg_mine_bienc_model exists = {os.path.isfile(self.config.neg_mine_bienc_model_file)} ")
				return None
				
		elif self.config.model_type == "bi_enc":
			# Use model only if epoch > 0 or pretrained model was specified in self.config.path_to_model
			if self.trainer.current_epoch > 0 or os.path.isfile(self.config.path_to_model):
				LOGGER.info(f"Returning current biencoder model")
				return self.trainer.model
			else:
				LOGGER.info(f"Returning None for biencoder model as current_epoch>0 ->{self.trainer.current_epoch > 0},"
							f" path_to_model file exists -> {os.path.isfile(self.config.path_to_model)}")
				return None
		else:
			raise Exception(f"Model type = {self.config.model_type} {type(self.trainer.model)} not supported")
		
			

class BasePairwiseTrainer(object):
	"""
	Trainer class to train a (pairwise) similarity model
	"""
	def __init__(self, config):
		
		assert isinstance(config, Config)
		self.config = config
		
		LOGGER.addHandler(logging.FileHandler(f"{self.config.result_dir}/log_file.txt"))
		
		# wandb initialization
		config_dict = self.config.__dict__
		config_dict["CUDA_DEVICE"] = os.environ["CUDA_VISIBLE_DEVICES"]
		
		self.wandb_logger = WandbLogger(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict)
		try:
			wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict)
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(project=f"{self.config.exp_id}", dir=self.config.result_dir, config=config_dict, mode="offline")
		
		
		# Create datamodule
		self.data	  	= EntLinkData(config=config)
		
		# Load the model and optimizer
		self.pw_model 	= load_pairwise_model(config=self.config)
		
		
		
	def __str__(self):
		print_str = pprint.pformat(self.config.to_json())
		print_str += f"Model parameters:{self.pw_model}"
		return print_str
		
		
	def train(self):
		seed_everything(self.config.seed, workers=True)

		if self.config.model_type == "bi_enc" and self.config.neg_strategy in ["bienc_hard_negs", "top_ce_w_bienc_hard_negs_trp"]:
			assert self.config.reload_dataloaders_every_n_epochs == 1, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}"
		else:
			assert self.config.reload_dataloaders_every_n_epochs == 0, f"Invalid combo of model_type = {self.config.model_type} and neg_strategy = {self.config.neg_strategy}"
		
		metric_to_monitor =  f"dev_{self.config.ckpt_metric}"
		checkpoint_callback = ModelCheckpoint(
			save_top_k=self.config.num_top_k_ckpts,
			monitor=metric_to_monitor,
			mode="min",
			auto_insert_metric_name=False,
			save_last=False, # Determines if we save another copy of checkpoint
			save_weights_only=False,
			dirpath=self.config.model_dir,
			filename="model-{epoch}-{train_step}-{" + metric_to_monitor + ":.2f}",
			save_on_train_epoch_end=False # Set to False to run check-pointing checks at the end of the val loop
		)
		checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
		
		
		end_of_epoch_checkpoint_callback = ModelCheckpoint(
			auto_insert_metric_name=False,
			save_last=False,
			save_weights_only=False,
			dirpath=self.config.model_dir,
			filename="eoe-{epoch}-last",
			save_on_train_epoch_end=True # Set to True to run check-pointing checks at end of each epoch
		)
		end_of_epoch_checkpoint_callback.CHECKPOINT_NAME_LAST = "eoe-{epoch}-last"
		
		
		lr_monitor = LearningRateMonitor(logging_interval='step')
		strategy = self.config.strategy if self.config.strategy in ["dp", "ddp", "ddp_spawn"] else None
		assert self.config.num_gpus <= 1 or strategy is not None, f"Can not pass more than 1 GPU with strategy = {strategy}"
		
		auto_lr_find = False
		
		val_check_interval = int(self.config.eval_interval*self.config.grad_acc_steps) if self.config.eval_interval > 1 else self.config.eval_interval
		trainer = Trainer(
			gpus=self.config.num_gpus,
			strategy=strategy,
			default_root_dir=self.config.result_dir,
			max_epochs=self.config.num_epochs,
			max_time=self.config.max_time,
			accumulate_grad_batches=self.config.grad_acc_steps,
			reload_dataloaders_every_n_epochs=self.config.reload_dataloaders_every_n_epochs,
			fast_dev_run=self.config.fast_dev_run,
			val_check_interval=val_check_interval,
			gradient_clip_val=self.config.max_grad_norm,
			callbacks=[lr_monitor, checkpoint_callback, end_of_epoch_checkpoint_callback],
			logger=self.wandb_logger,
			auto_lr_find=auto_lr_find,
			profiler="simple",
			num_sanity_val_steps=0
		)
			
		ckpt_path = self.config.ckpt_path if self.config.ckpt_path != "" else None
		trainer.fit(model=self.pw_model, datamodule=self.data, ckpt_path=ckpt_path)
