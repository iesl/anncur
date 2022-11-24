import os
import sys
import copy
import torch
import logging
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
import pytorch_lightning as pl

from eval.eval_utils import score_topk_preds
from models.biencoder import BertWrapper
from models.params import ENT_TITLE_TAG, ENT_START_TAG, ENT_END_TAG
from utils.config import Config
from utils.optimizer import get_bert_optimizer, get_scheduler

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def to_cross_bert_input(token_idxs, null_idx, first_segment_end):
	"""
	This function is used for preparing input for a cross-encoder bert model
	Create segment_idx and mask tensors for feeding the input to BERTs

	:param token_idxs: is a 2D int tensor.
	:param null_idx: idx of null element
	:param first_segment_end: idx where next segment i.e. label starts in the token_idxs tensor
	:return: return token_idx, segment_idx and mask
	"""
	# TODO: Verify that this is behaving as expected. Segment_idxs should be correct.
	segment_idxs = token_idxs * 0
	if first_segment_end > 0:
		segment_idxs[:, first_segment_end:] = token_idxs[:, first_segment_end:] > 0
	
	mask = token_idxs != null_idx
	# nullify elements in case self.NULL_IDX was not 0
	token_idxs = token_idxs * mask.long()
	
	return token_idxs, segment_idxs, mask


class CrossBertWEmbedsWrapper(nn.Module):
	"""
	Wrapper around BERT model which is used as a cross-encoder model.
	This first estimates contextualized embeddings for each input and then outputs score for the given input.
	"""
	def __init__(self, bert_model, pooling_type, bert_model_type='bert-base-uncased'):
		super(CrossBertWEmbedsWrapper, self).__init__()

		self.bert_model = bert_model
		
		self.pooling_type = pooling_type # TODO: Remove this param?
		self.bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)
		
		# TODO: Remove hardcoded do_lower_case=True here
		tokenizer = BertTokenizer.from_pretrained(bert_model_type, do_lower_case=True)
		self.ENT_START_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_START_TAG)
		self.ENT_END_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_END_TAG)
		self.ENT_TITLE_TAG_ID = tokenizer.convert_tokens_to_ids(ENT_TITLE_TAG)
		

	def forward(self, token_ids, segment_ids, attention_mask):
		input_1_embed, input_2_embed  = self.forward_for_embeds(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask
		)
		output_scores = torch.sum(input_1_embed*input_2_embed, dim=-1) # Shape: (batch_size, )
		
		# FIXME: Adding extra dim as wrapper around this class would call squeeze() before returning final scores, Remove this
		output_scores = output_scores.unsqueeze(1)
		return output_scores
	
	
	def forward_for_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return: returns two embedding tensors of shape (batch_size, embed_dim)
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		elif len(output) == 4:
			output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_start_tag_idxs = (token_ids == self.ENT_START_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_end_tag_idxs = (token_ids == self.ENT_END_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_title_idxs = (token_ids == self.ENT_TITLE_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_startm ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_start_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_start_tag_idxs.shape = {ent_start_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_end_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_end_tag_idxs.shape = {ent_end_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_title_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_title_idxs.shape = {ent_title_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_title_embeds = torch.stack([output_bert[i, ent_title_idxs[i, 1], :] for i in range(batch_size)])
		
		ent_start_embeds = torch.stack([output_bert[i, ent_start_tag_idxs[i, 1], :] for i in range(batch_size)])
		ent_end_embeds = torch.stack([output_bert[i, ent_end_tag_idxs[i, 1], :] for i in range(batch_size)])
		
		# LOGGER.info(f"ent_title_embeds.shape  = {ent_title_embeds.shape}")
		# LOGGER.info(f"ent_title_embeds_list.shape  = {ent_title_embeds_list[0].shape}")
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		input_1_embed = (ent_start_embeds + ent_end_embeds)/2 # shape: (batch_size, bert_output_dim)
		input_2_embed = ent_title_embeds # shape: (batch_size, bert_output_dim)
		
		return input_1_embed, input_2_embed
	
	
	def forward_for_input_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		Only extract embedding for input (eg mention in entity linking)
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return:
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_start_tag_idxs = (token_ids == self.ENT_START_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		ent_end_tag_idxs = (token_ids == self.ENT_END_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_start, ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_start_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_start_tag_idxs.shape = {ent_start_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		assert ent_end_tag_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_end_tag_idxs.shape = {ent_end_tag_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_start_embeds = torch.stack([output_bert[i, ent_start_tag_idxs[i, 1], :] for i in range(batch_size)])
		ent_end_embeds = torch.stack([output_bert[i, ent_end_tag_idxs[i, 1], :] for i in range(batch_size)])
		
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		input_embed = (ent_start_embeds + ent_end_embeds)/2 # shape: (batch_size, bert_output_dim)
		
		return input_embed
	
	
	def forward_for_label_embeds(self, token_ids, segment_ids, attention_mask):
		"""
		Only extract embedding for label (eg entity in entity linking)
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:return:
		"""
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		# elif len(output) == 4:
		# 	output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		ent_title_idxs = (token_ids == self.ENT_TITLE_TAG_ID).nonzero() # shape : ( num_matches, len(token_ids.shape()) )
		
		# TODO Assert that there is at max one ent_start, ent_end, and ent_title token in each sequence in batch
		batch_size = token_ids.shape[0]
		assert len(token_ids.shape) == 2, f"len(token_ids.shape) = {len(token_ids.shape)} != 2"
		assert ent_title_idxs.shape == (batch_size, len(token_ids.shape)), f"Shape mismatch ent_title_idxs.shape = {ent_title_idxs.shape} != (batch_size, len(token_ids.shape)) = {(batch_size, len(token_ids.shape))}"
		
		# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
		# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
		ent_title_embeds = torch.stack([output_bert[i, ent_title_idxs[i, 1], :] for i in range(batch_size)])
		
		# For each input seq in batch, figure out how to compute their embeddings from this sequence of contextualized embeddings
		label_embeds = ent_title_embeds # shape: (batch_size, bert_output_dim)
		
		return label_embeds
		
	
class CrossBertWrapper(BertWrapper):
	"""
	Wrapper around BERT model which is used as a cross-encoder model. This wrapper outputs scores for the given input.
	"""
	def __init__(self, bert_model, pooling_type):
		super(CrossBertWrapper, self).__init__(bert_model=bert_model,
											   output_dim=1,
											   pooling_type=pooling_type,
											   add_linear_layer=True)
		

	

	def forward(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		scores = super(CrossBertWrapper, self).forward(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			pooling_type=pooling_type
		)
		
		return scores


class CrossEncoderModule(torch.nn.Module):
	"""
	This class enables interfacing with different cross-encoder architectures
	(such as CrossBertWrapper and CrossBertWEmbedsWrapper) for scoring input pairs.
	"""
	def __init__(self, bert_model, pooling_type, bert_args, cross_enc_type="default"):
		super(CrossEncoderModule, self).__init__()
		
		cross_bert = BertModel.from_pretrained(bert_model, **bert_args) # BERT Model for cross encoding input and labels
		
		self.bert_config = cross_bert.config
		
		if cross_enc_type == "default":
			self.encoder = CrossBertWrapper(
				bert_model=cross_bert,
				pooling_type=pooling_type
			)
		elif cross_enc_type == "w_embeds":
			self.encoder = CrossBertWEmbedsWrapper(
				bert_model=cross_bert,
				pooling_type=pooling_type,
				bert_model_type=bert_model
			)
		else:
			raise Exception(f"CrossEncoder type = {cross_enc_type} not supported")
		

	def forward(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		embedding = self.encoder(token_idx, segment_idx, mask)
		return embedding.squeeze(-1) # Remove last dim
	
	
	def forward_for_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		embeddings_1, embeddings_2  = self.encoder.forward_for_embeds(token_idx, segment_idx, mask)
		return embeddings_1, embeddings_2
	
	def forward_for_input_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		if isinstance(self.encoder, CrossBertWEmbedsWrapper):
			return self.encoder.forward_for_input_embeds(token_idx, segment_idx, mask)
		elif isinstance(self.encoder, CrossBertWrapper):
			return self.encoder.forward_wo_linear(token_idx, segment_idx, mask)
		else:
			raise NotImplementedError(f"encoder of type={type(self.encoder)} not supported")
	
	def forward_for_label_embeds(
		self,
		token_idx,
		segment_idx,
		mask,
	):
		if isinstance(self.encoder, CrossBertWEmbedsWrapper):
			return self.encoder.forward_for_label_embeds(token_idx, segment_idx, mask)
		elif isinstance(self.encoder, CrossBertWrapper):
			return self.encoder.forward_wo_linear(token_idx, segment_idx, mask)
		else:
			raise NotImplementedError(f"encoder of type={type(self.encoder)} not supported")
		
	
class CrossEncoderWrapper(pl.LightningModule):
	"""
	This class adds functionality of computing loss for training cross-encoder models
	"""
	def __init__(self, config):
		super(CrossEncoderWrapper, self).__init__()
		assert isinstance(config, Config)
		
		self.config = config
		self.learning_rate = self.config.learning_rate
		
		# init tokenizer
		self.tokenizer = BertTokenizer.from_pretrained(
			self.config.bert_model, do_lower_case=self.config.lowercase
		)
		self.NULL_IDX = self.tokenizer.pad_token_id
		
		# init model
		self.model = self.build_encoder_model()
		
		# Load parameters from file if it exists
		if os.path.exists(self.config.path_to_model):
			LOGGER.info(f"Loading parameters from {self.config.path_to_model}")
			self.update_encoder_model(skeleton_model=self.model, fname=self.config.path_to_model)
		else:
			LOGGER.info(f"Running with default parameters as self.config.path_to_model = {self.config.path_to_model} does not exist")

		# Move model to appropriate device
		self.to(self.config.device)
		self.model = self.model.to(self.device)
		self.save_hyperparameters()
		try:
			LOGGER.info(f"Model device is {self.device} {self.config.device}")
		except:
			pass
	
	@property
	def model_config(self):
		if isinstance(self.model, CrossEncoderModule):
			return self.model.bert_config
		elif isinstance(self.model, torch.nn.parallel.DataParallel) and isinstance(self.model.module, CrossEncoderModule):
			return self.model.module.bert_config
		else:
			raise Exception(f"model type = {type(self.model)} does not have a model config")
	
	@classmethod
	def load_model(cls, config):
		"""
		Load parameters from config file and create an object of this class
		:param config:
		:return:
		"""
		if isinstance(config, str):
			with open(config, "r") as f:
				return cls(Config(f))
		elif isinstance(config, Config):
			return cls(config)
		elif isinstance(config, dict):
			config_obj = Config()
			config_obj.__dict__.update(config)
			return cls(config_obj)
		else:
			raise Exception(f"Invalid config param = {config}")
	
	def save_model(self, res_dir):
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		
		self.config.encoder_wrapper_config = os.path.join(res_dir, "wrapper_config.json")
		self.save_encoder_model(res_dir=res_dir)
		self.config.save_config(res_dir=res_dir, filename="wrapper_config.json")
		
	def build_encoder_model(self):
		"""
		Build an (optionally pretrained) encoder model with the desired architecture.
		:return:
		"""
		cross_enc_type = self.config.cross_enc_type if hasattr(self.config, "cross_enc_type") else "default"
		bert_args = copy.deepcopy(self.config.bert_args)
		
		return CrossEncoderModule(
			bert_model=self.config.bert_model,
			pooling_type=self.config.pooling_type,
			bert_args=bert_args,
			cross_enc_type=cross_enc_type
		)
	
	def save_encoder_model(self, res_dir):
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		
		model_file = os.path.join(res_dir, "model.torch")
		LOGGER.info("Saving encoder model to :{}".format(model_file))
		
		self.config.path_to_model = model_file
		if isinstance(self.model, torch.nn.DataParallel):
			torch.save(self.model.module.state_dict(), model_file)
		else:
			torch.save(self.model.state_dict(), model_file)
		
		
		model_config_file = os.path.join(res_dir, "model.config")
		self.model_config.to_json_file(model_config_file)
		
		self.tokenizer.save_vocabulary(res_dir)
	
	@staticmethod
	def update_encoder_model(skeleton_model, fname, cpu=False):
		"""
		Read state_dict in file fname and load parameters into skeleton_model
		:param skeleton_model: Model with the desired architecture
		:param fname:
		:param cpu:
		:return:
		"""
		if cpu:
			state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
		else:
			state_dict = torch.load(fname)
		if 'state_dict' in state_dict: # Load for pytorch lightning checkpoint
			model_state_dict = {}
			for key,val in state_dict['state_dict'].items():
				if key.startswith("model."):
					model_state_dict[key[6:]] = val
				else:
					model_state_dict[key] = val
					
			skeleton_model.load_state_dict(model_state_dict)
		else:
			skeleton_model.load_state_dict(state_dict)
	
	def encode(self, token_idxs, enc_to_use, first_segment_end):
		
		token_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=token_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		if enc_to_use == "input":
			return self.model.forward_for_input_embeds(
				token_idx=token_idxs,
				segment_idx=segment_idxs,
				mask=mask,
			)
		elif enc_to_use == "label":
			return self.model.forward_for_label_embeds(
				token_idx=token_idxs,
				segment_idx=segment_idxs,
				mask=mask,
			)
		else:
			raise NotImplementedError(f"Enc_to_use = {enc_to_use} not supported")
	
	def encode_input(self, input_token_idxs, first_segment_end=0):
		return self.encode(token_idxs=input_token_idxs, enc_to_use="input", first_segment_end=first_segment_end)

	def encode_label(self, label_token_idxs, first_segment_end=0):
		return self.encode(token_idxs=label_token_idxs, enc_to_use="label", first_segment_end=first_segment_end)
		

	# Score candidates given context input and label input
	def score_paired_input_and_labels(self, input_pair_idxs, first_segment_end):
		orig_shape = input_pair_idxs.shape
		input_pair_idxs = input_pair_idxs.view(-1, orig_shape[-1])
		
		# input_idxs.shape : batch_size x max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		
		# Score the pairs
		scores = self.model(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size,)

		scores = scores.view(orig_shape[:-1]) # Convert to shape compatible with original input shape
		return scores
		
		
	def score_candidate(self, input_pair_idxs, first_segment_end):
		return self.score_paired_input_and_labels(input_pair_idxs=input_pair_idxs, first_segment_end=first_segment_end)
	
	
	def embed_paired_input_and_labels(self, input_pair_idxs, first_segment_end):
		orig_shape = input_pair_idxs.shape
		input_pair_idxs = input_pair_idxs.view(-1, orig_shape[-1])
		
		# input_idxs.shape : batch_size x max_num_tokens
		# Prepare input_idxs for feeding into bert model
		input_pair_idxs, segment_idxs, mask = to_cross_bert_input(
			token_idxs=input_pair_idxs, null_idx=self.NULL_IDX, first_segment_end=first_segment_end,
		)
		
		# Get embeddings for inputs and labels -
		input_embeds, label_embeds = self.model.forward_for_embeds(input_pair_idxs, segment_idxs, mask,) # Shape: (batch_size, embed_dim)

		return input_embeds, label_embeds
	

	def forward(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		loss, pos_scores, neg_scores = self.forward_w_scores(
			pos_pair_idxs=pos_pair_idxs,
			neg_pair_idxs=neg_pair_idxs,
			first_segment_end=first_segment_end
		)
		return loss
		
	def forward_w_scores(self, pos_pair_idxs, neg_pair_idxs, first_segment_end):
		
		pos_scores = self.score_paired_input_and_labels(
			input_pair_idxs=pos_pair_idxs,
			first_segment_end=first_segment_end
		) # (batch_size, )
		
		batch_size, num_negs, seq_len = neg_pair_idxs.shape
		neg_scores = self.score_paired_input_and_labels(
			input_pair_idxs=neg_pair_idxs.view(batch_size*num_negs, seq_len),
			first_segment_end=first_segment_end
		) # (batch_size*num_negs, 1)
		
		neg_scores = neg_scores.view(batch_size, num_negs) # (batch_size, num_negs)
		
		loss = self.compute_loss_w_scores(
			pos_scores=pos_scores,
			neg_scores=neg_scores
		)
		return loss, pos_scores, neg_scores
	
	def compute_loss_w_scores(self, pos_scores, neg_scores):
		"""
		Compute various losses given scores for pos and neg labels
		:param pos_scores: Tensor of shape (batch_size, )
		:param neg_scores: Tensor of shape (batch_size, num_negs)
		:return:
		"""
		
		
		if self.config.loss_type == "bce":
			loss = self.compute_binary_cross_ent_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores
			)
			return loss
		elif self.config.loss_type == "ce":
			loss = self.compute_cross_ent_loss(
				pos_scores=pos_scores,
				neg_scores=neg_scores
			)
			return loss
		else:
			raise NotImplementedError(f"Loss function of type = {self.config.loss_type} not implemented")
		
	@staticmethod
	def compute_eval_metrics(pos_scores, neg_scores):
		"""
		
		:param pos_scores: score tensor of shape (batch_size,)
		:param neg_scores: score tensor of shape (batch_size, num_negs)
		:return:
		"""
		batch_size, num_negs = neg_scores.shape
		
		batch_size = pos_scores.shape[0]
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row in final_scores contained score for positive label
		target = np.zeros(batch_size)
		
		# Sort scores based on preds
		topk_scores, topk_indices = final_scores.topk(k=num_negs+1)
		topk_preds = {"indices":topk_indices.cpu().detach().numpy(),
					  "scores":topk_scores.cpu().detach().numpy()}
		
		res_metrics = score_topk_preds(gt_labels=target, topk_preds=topk_preds)
		
		return res_metrics
	
	@staticmethod
	def compute_binary_cross_ent_loss(pos_scores, neg_scores):
		"""
		Compute binary cross-entropy loss for each pos and neg score
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:return:
		"""
		pos_target = torch.ones(pos_scores.shape, device=pos_scores.device)
		neg_target = torch.zeros(neg_scores.shape, device=neg_scores.device)
		
		pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_target, reduction="mean")
		neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_target, reduction="mean")
		
		loss = (pos_loss + neg_loss)/2
		
		return loss
	
	@staticmethod
	def compute_cross_ent_loss(pos_scores, neg_scores):
		"""
		Compute cross-entropy loss
		:param pos_scores: (batch_size,) dim tensor with scores for each pos-label
		:param neg_scores: (batch_size, num_neg) dim tensor with scores for neg-label for each input in batch
		:return:
		"""
		
		batch_size = pos_scores.shape[0]
		assert len(pos_scores.shape) == 1
		
		pos_scores = pos_scores.unsqueeze(1) # Add a dim to concatenate with neg_scores : Shape (batch_size, 1)
		
		final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
		
		# 0th col in each row in final_scores contained score for positive label
		target = torch.zeros((batch_size), dtype=torch.long, device=final_scores.device)
		
		loss = F.cross_entropy(final_scores, target, reduction="mean")
		return loss
		
		
	def configure_optimizers(self):
		optimizer = get_bert_optimizer(
			models=[self],
			type_optimization=self.config.type_optimization,
			learning_rate=self.learning_rate,
			weight_decay=self.config.weight_decay,
			optimizer_type="AdamW"
		)
		# len_data = len(self.trainer._data_connector._train_dataloader_source.instance)
		len_data = self.trainer.datamodule.train_data_len
		# len_data is already adjusted taking into batch_size and grad_acc_steps into account so pass 1 for these
		scheduler = get_scheduler(
			optimizer=optimizer,
			epochs=self.config.num_epochs,
			warmup_proportion=self.config.warmup_proportion,
			len_data=len_data,
			batch_size=1,
			grad_acc_steps=1
		)
		
		lr_scheduler_config = {
			"scheduler": scheduler,
			"interval": "step",
			"frequency": 1,
		}
		return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}
	
	def training_step(self, train_batch, batch_idx):
		
		assert len(train_batch) == 2, f"Number of elements in train_batch = {len(train_batch)} is not supported"
		batch_pos_pairs, batch_neg_pairs = train_batch
		
		return self.forward(
			pos_pair_idxs=batch_pos_pairs,
			neg_pair_idxs=batch_neg_pairs,
			first_segment_end=self.config.max_input_len
		)
		
	def validation_step(self, val_batch, batch_idx):
		
		assert len(val_batch) == 2, f"Number of elements in val_batch = {len(val_batch)} is not supported"
		batch_pos_pairs, batch_neg_pairs = val_batch
		res_metrics = {}
		loss, pos_scores, neg_scores = self.forward_w_scores(
			pos_pair_idxs=batch_pos_pairs,
			neg_pair_idxs=batch_neg_pairs,
			first_segment_end=self.config.max_input_len
		)
		res_metrics["loss"] = loss
		
		if self.config.ckpt_metric == "mrr":
			temp_metrics = self.compute_eval_metrics(
				pos_scores=pos_scores,
				neg_scores=neg_scores,
			)
			res_metrics.update(temp_metrics)
			return res_metrics
		elif self.config.ckpt_metric == "loss":
			return res_metrics
		else:
			raise NotImplementedError(f"ckpt metric = {self.config.ckpt_metric} not supported")
		
	def validation_epoch_end(self, outputs):
		
		super(CrossEncoderWrapper, self).validation_epoch_end(outputs=outputs)
		
		eval_loss = torch.mean(torch.tensor([scores["loss"] for scores in outputs])) # Avg loss numbers
		
		if self.config.ckpt_metric == "mrr":
			eval_metric = np.mean([float(scores["mrr"]) for scores in outputs]) # Avg MRR numbers
			# Usually we use loss for eval_metric and want to find params that minimize this loss.
			# Since higher MRR is better, we multiply eval_metric with -1 so that we can still use min of this metric for checkpointing purposes
			eval_metric = -1*eval_metric
			
			self.log(f"dev_loss", eval_loss, sync_dist=True, on_epoch=True, logger=True)
			
		elif self.config.ckpt_metric == "loss":
			eval_metric = eval_loss
		else:
			raise NotImplementedError(f"ckpt metric = {self.config.ckpt_metric} not supported")
		
		self.log(f"dev_{self.config.ckpt_metric}", eval_metric, sync_dist=True, on_epoch=True, logger=True)
		
	def on_train_start(self):
		super(CrossEncoderWrapper, self).on_train_start()
		LOGGER.info("On Train Start")
		self.log("train_step", self.global_step, logger=True)
	
	def on_train_epoch_start(self):
		
		super(CrossEncoderWrapper, self).on_train_epoch_start()
		if self.config.reload_dataloaders_every_n_epochs and (self.current_epoch % self.config.reload_dataloaders_every_n_epochs == 0)\
				and self.trainer.checkpoint_callback:
			LOGGER.info(f"\n\n\t\tResetting model checkpoint callback params in epoch = {self.current_epoch}\n\n")
			for checkpoint_callback in self.trainer.checkpoint_callbacks:
				checkpoint_callback.current_score = None
				checkpoint_callback.best_k_models = {}
				checkpoint_callback.kth_best_model_path = ""
				checkpoint_callback.best_model_score = None
				checkpoint_callback.best_model_path = ""
				checkpoint_callback.last_model_path = ""
			
	def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
		super(CrossEncoderWrapper, self).on_train_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx, unused=unused)
		
		if (self.global_step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
			
			self.log("train_loss", outputs, on_epoch=True, logger=True)
			self.log("train_step", self.global_step, logger=True)
			self.log("train_epoch", self.current_epoch, logger=True)
