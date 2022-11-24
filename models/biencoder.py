import os
import sys
import torch
import logging
import itertools

import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_lightning import LightningModule

from utils.config import Config
from utils.optimizer import get_bert_optimizer, get_scheduler

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def to_bert_input(token_idxs, null_idx):
	"""
	Create segment_idx and mask tensors for feeding the input to BERTs

	:param token_idxs: is a 2D int tensor.
	:param null_idx: idx of null element
	:return: return token_idx, segment_idx and mask
	"""
	
	segment_idx = token_idxs * 0
	mask = token_idxs != null_idx
	# nullify elements in case self.NULL_IDX was not 0
	token_idxs = token_idxs * mask.long()
	return token_idxs, segment_idx, mask
	

class BertWrapper(nn.Module):
	
	"""
	Wrapper class around BERT model. This enables using different pooling methods to get representation of
	the input from BERT model and optionally adding additional linear layer on top of representation from BERT
	and
	"""
	def __init__(self, bert_model, pooling_type, output_dim, add_linear_layer):
		super(BertWrapper, self).__init__()
		self.bert_model = bert_model
		
		self.add_linear_layer = add_linear_layer
		self.output_dim = output_dim
		self.pooling_type = pooling_type
		self.bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

		# Either output_dim should match bert_output_dim or we should be adding a linear layer
		assert (self.output_dim == self.bert_output_dim) or self.add_linear_layer, f"output_dim = {self.output_dim}, and bert_output_dim = {self.bert_output_dim}. Since no linear layer is added on top of bert output, these two values should match"
		
		if self.add_linear_layer:
			self.dropout = nn.Dropout(0.1)
			self.additional_linear = nn.Linear(self.bert_output_dim, self.output_dim)
			# self.additional_linear.weight.data.normal_(
			# 	mean=0.0, std=self.bert_model.config.initializer_range)
			# self.additional_linear.bias.data.zero_()
		else:
			self.additional_linear = None
			

	def forward(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		
		embeddings = self.forward_wo_linear(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			pooling_type=pooling_type
		)
		
		if self.additional_linear is not None:
			# get pooled output from BERT
			# and pass it through additional layer here.
			embeddings = self.additional_linear(self.dropout(embeddings))

		return embeddings
	
	def forward_wo_linear(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		"""
		Computes output for given input but DOES NOT pass it through final additional layer.
		
		:param token_ids:
		:param segment_ids:
		:param attention_mask:
		:param pooling_type:
		:return:
		"""
		if pooling_type is None:
			pooling_type = self.pooling_type
		
		output = self.bert_model(token_ids, segment_ids, attention_mask)
		if len(output) == 2:
			output_bert, output_pooler = output
		elif len(output) == 4:
			output_bert, output_pooler, all_hidden_units, all_attention_weights  = output
		else:
			raise Exception(f"Unexpected number of values in output = {len(output)}")
			
		if pooling_type == "cls_w_lin": # cls token embedding which is passed through linear layer in BERT model itself
			embeddings = output_pooler
		elif pooling_type == "cls": # Use exact cls token embedding
			# output_bert shape = (batch_size, max_seq_len, bert_output_dim)
			# output_bert[:, 0, :] -> (batch_size, bert_output_dim) tensor with values for first token in each sequence in the batch
			embeddings = output_bert[:, 0, :]
		elif pooling_type == "mean":
			embeddings = torch.mean(output_bert, dim=1) # Pool along dim = 1 so as to average along seq_len dimension
		elif pooling_type == "max":
			embeddings = torch.max(output_bert, dim=1) # Pool along dim = 1 so as to average along seq_len dimension
		elif pooling_type == "lse": # Log-Sum-Exp pooling
			embeddings = torch.logsumexp(output_bert, dim=1) # Pool along dim = 1 so as to average along seq_len dimension
		elif pooling_type == "spl_tkns":
			raise NotImplementedError
		else:
			# embeddings = None
			raise NotImplementedError("Pooling type = {} not supported yet")
		
		# embeddings.shape  == (batch_size, bert_output_dim)
		assert embeddings.shape == (output_bert.shape[0], self.bert_output_dim)
		
		return embeddings
	
	
	def forward_for_input_embeds(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		return self.forward_wo_linear(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			pooling_type=pooling_type
		)
	
	def forward_for_label_embeds(self, token_ids, segment_ids, attention_mask, pooling_type=None):
		return self.forward_wo_linear(
			token_ids=token_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			pooling_type=pooling_type
		)


class BiEncoderModule(torch.nn.Module):
	"""
	Module to enable various ways of parameterizing query and item encoder in a bi-encoder model
	"""
	def __init__(self, bert_model, embed_dim, pooling_type, add_linear_layer, bi_enc_type="separate"):
		super(BiEncoderModule, self).__init__()
		from models.crossencoder import CrossBertWEmbedsWrapper
		
		self.bi_enc_type = bi_enc_type
		self.pooling_type = pooling_type
		if bi_enc_type == "separate":
			# assert pooling_type != "spl_tkns", f"Pooling type = {pooling_type} not supported with bi_enc_type={bi_enc_type}"
			input_bert = BertModel.from_pretrained(bert_model) # BERT Model for encoding input
			label_bert = BertModel.from_pretrained(bert_model) # BERT Model for encoding labels
			self.bert_config = input_bert.config
			
			if pooling_type == "spl_tkns":
				self.input_encoder = CrossBertWEmbedsWrapper(
					bert_model=input_bert,
					pooling_type=pooling_type,
				)
				self.label_encoder = CrossBertWEmbedsWrapper(
					bert_model=label_bert,
					pooling_type=pooling_type
				)
			else:
				self.input_encoder = BertWrapper(
					bert_model=input_bert,
					output_dim=embed_dim,
					pooling_type=pooling_type,
					add_linear_layer=add_linear_layer
					
				)
				self.label_encoder = BertWrapper(
					bert_model=label_bert,
					output_dim=embed_dim,
					pooling_type=pooling_type,
					add_linear_layer=add_linear_layer
				)
		elif bi_enc_type == "shared":
			
			if pooling_type == "spl_tkns":
				
				encoder_bert = BertModel.from_pretrained(bert_model) # BERT Model for encoding input and labels
			
				self.bert_config = encoder_bert.config
			
				self.encoder = CrossBertWEmbedsWrapper(
					bert_model=encoder_bert,
					pooling_type=pooling_type,
				)
			else:
				encoder_bert = BertModel.from_pretrained(bert_model) # BERT Model for encoding input and labels
			
				self.bert_config = encoder_bert.config
			
				self.encoder = BertWrapper(
					bert_model=encoder_bert,
					output_dim=embed_dim,
					pooling_type=pooling_type,
					add_linear_layer=add_linear_layer
				)
			# self.label_encoder = self.encoder
			# self.input_encoder = self.encoder
		else:
			raise NotImplementedError(f"bi_enc_type = {bi_enc_type} not supported")
			
		

	def forward(
		self,
		token_idx,
		segment_idx,
		mask,
		enc_to_use
	):
		from models.crossencoder import CrossBertWEmbedsWrapper
		if self.bi_enc_type == "separate" and isinstance(self.label_encoder, BertWrapper) and isinstance(self.input_encoder, BertWrapper):
			if enc_to_use == "input":
				return self.input_encoder(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			elif enc_to_use == "label":
				return self.label_encoder(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			else:
				raise Exception(f"enc_to_use = {enc_to_use} not supported")
		elif self.bi_enc_type == "separate" and isinstance(self.label_encoder, CrossBertWEmbedsWrapper) and isinstance(self.input_encoder, CrossBertWEmbedsWrapper):
			if enc_to_use == "input":
				return self.input_encoder.forward_for_input_embeds(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			elif enc_to_use == "label":
				return self.label_encoder.forward_for_label_embeds(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			else:
				raise Exception(f"enc_to_use = {enc_to_use} not supported")
		elif self.bi_enc_type == "shared" and isinstance(self.encoder, BertWrapper):
			return self.encoder(
				token_ids=token_idx,
				segment_ids=segment_idx,
				attention_mask=mask
			)
		elif self.bi_enc_type == "shared" and isinstance(self.encoder, CrossBertWEmbedsWrapper):
			if enc_to_use == "input":
				return self.encoder.forward_for_input_embeds(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			elif enc_to_use == "label":
				return self.encoder.forward_for_label_embeds(
					token_ids=token_idx,
					segment_ids=segment_idx,
					attention_mask=mask
				)
			else:
				raise Exception(f"enc_to_use = {enc_to_use} not supported")
		else:
			raise NotImplementedError(f"bi_enc_type = {self.bi_enc_type} and "
									  f"encoder types = {type(self.input_encoder) if hasattr(self, 'input_encoder') else type(self.encoder)}, "
									  f"{type(self.label_encoder) if hasattr(self, 'label_encoder') else type(self.encoder)} not supported")


class BiEncoderWrapper(LightningModule):
	"""
	PyTorch Lightning Module with loss function for training a Bi-Encoder Model
	"""
	def __init__(self, config):
		super(BiEncoderWrapper, self).__init__()
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
			LOGGER.info(f"Updating to parameters from model={self.config.path_to_model}")
			self.update_encoder_model(skeleton_model=self.model, fname=self.config.path_to_model)
		else:
			LOGGER.info(f"Running with default parameters as path_to_model={self.config.path_to_model} does not exist")

		# Move model to appropriate device (No need with pytorch lightning)
		self.to(self.config.device)
		self.model = self.model.to(self.device)

		self.save_hyperparameters()
	
	@property
	def model_config(self):
		if isinstance(self.model, BiEncoderModule):
			return self.model.bert_config
		elif isinstance(self.model, torch.nn.parallel.DataParallel) and isinstance(self.model.module, BiEncoderModule):
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
		
		bi_enc_type = self.config.bi_enc_type if hasattr(self.config, "bi_enc_type") else "separate"
		return BiEncoderModule(
			bert_model=self.config.bert_model,
			embed_dim=self.config.embed_dim,
			pooling_type=self.config.pooling_type,
			add_linear_layer=self.config.add_linear_layer,
			bi_enc_type=bi_enc_type
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

	
	def encode(self, token_idxs, enc_to_use):
		token_idxs, segment_idxs, masks = to_bert_input(token_idxs=token_idxs, null_idx=self.NULL_IDX)
		embedding = self.model(
			token_idx=token_idxs,
			segment_idx=segment_idxs,
			mask=masks,
			enc_to_use=enc_to_use
		)
		
		return embedding
	
	def encode_input(self, input_token_idxs):
		return self.encode(token_idxs=input_token_idxs, enc_to_use="input")

	def encode_label(self, label_token_idxs):
		return self.encode(token_idxs=label_token_idxs, enc_to_use="label")
		
	def encode_context(self, token_idxs):
		return self.encode_input(input_token_idxs=token_idxs)
	
	def encode_candidate(self, token_idxs):
		return self.encode_label(label_token_idxs=token_idxs)
	
	
	def score_labels(
		self,
		input_token_idxs,
		label_token_idxs,
		comp_all_scores=True,
	):
		# Encode all inputs in the batch.
		embedding_inputs = self.encode(token_idxs=input_token_idxs, enc_to_use="input")

		# # Candidate encoding is given, do not need to re-compute
		# # Directly return the score of context encoding and candidate encoding
		# if label_embeds is not None:
		# 	return embedding_inputs.mm(label_embeds.t())

		# Encode all labels in the batch.
		embedding_labels = self.encode(token_idxs=label_token_idxs, enc_to_use="label")
		
		if comp_all_scores:
			# Compute scores of cross-product of all inputs and labels in batch
			scores = embedding_inputs.mm(embedding_labels.t()) # Shape : (batch_size, batch_size)
			return scores
		else:
			# Compute scores of inputs with corresponding labels
			embedding_inputs = embedding_inputs.unsqueeze(1)  # batch_size x 1 x embed_size
			embedding_labels = embedding_labels.unsqueeze(2)  # batch_size x embed_size x 1
			scores = torch.bmm(embedding_inputs, embedding_labels)  # batch_size x 1 x 1
			scores = torch.squeeze(scores) # Shape : (batch_size,)
			return scores
		
	
	
	def forward(self, input_token_idxs, pos_label_token_idxs, neg_label_token_idxs):
		"""
		Embeds input and labels, and computes loss
		:param input_token_idxs: Shape (batch_size, seq_len) tensor with tokenized input
		:param pos_label_token_idxs: Shape (batch_size, seq_len) tensor with tokenized pos label for corresponding input
		:param neg_label_token_idxs: Shape (batch_size, num_negs_per_input, seq_len) tensor with tokenized neg labels
									for corresponding input
		
		:return:
		"""
		
		# Encode all inputs in the batch.
		input_embs = self.encode(token_idxs=input_token_idxs, enc_to_use="input") # (batch_size, embed_size)

		# Encode all positive labels in the batch.
		pos_label_embs = self.encode(token_idxs=pos_label_token_idxs, enc_to_use="label") # (batch_size, embed_size)
		
		if neg_label_token_idxs is not None: # Compute loss with given negatives for each input in batch
			batch_size, num_negs_per_input, label_seq_len = neg_label_token_idxs.shape
			neg_label_embs = self.encode(
				token_idxs=neg_label_token_idxs.view(batch_size*num_negs_per_input, label_seq_len),
				enc_to_use="label"
			) # Shape: (batch_size * num_negs_per_input, label_seq_len)
			
			# neg_label_token_idxs : Shape (batch_size, num_negs_per_input, label_seq_len)
			neg_label_embs = neg_label_embs.view(batch_size, num_negs_per_input, -1) # Shape: (batch_size,  num_negs_per_input, embed_size)
			
			loss = self.compute_loss_w_negs(
				input_embs=input_embs,
				pos_label_embs=pos_label_embs,
				neg_label_embs=neg_label_embs
			)
			
		else: # Compute loss with in-batch negatives
			# This is required to make in-batch negative computation easy and fast for multi-label classification datasets.
			# with data shuffling turned on, each batch is very less likely to consist of more than one positive label for
			# each input.
			assert self.config.shuffle_data or self.config.data_type != "xmc", "Shuffling should be turned on for XMC datasets to in-batch negatives mode"
			
			loss = self.compute_loss_w_in_batch_negs(
				input_embs=input_embs,
				pos_label_embs=pos_label_embs
			)
		
		return loss
	
	def forward_w_ment_ent_distill(self, input_token_idxs, label_token_idxs, target_label_scores):
		"""
		Embeds input and labels, and computes loss for using target_labels_scores. This is used for distillation
		experiments where both mention and entity encoders are trained.
		:param input_token_idxs: Shape (batch_size, seq_len) tensor with tokenized input
		:param label_token_idxs: Shape (batch_size, num_labels, seq_len) tensor with tokenized labels for corresponding input
		:param target_label_scores: Shape (batch_size, num_labels) tensor with scores for corresponding labels
		:return:
		"""
		# Encode all inputs in the batch.
		input_embs = self.encode(token_idxs=input_token_idxs, enc_to_use="input") # (batch_size, embed_size)

		batch_size, num_labels, label_seq_len = label_token_idxs.shape
		label_embs = self.encode(
			token_idxs=label_token_idxs.view(batch_size*num_labels, label_seq_len),
			enc_to_use="label"
		) # Shape: (batch_size * num_labels, label_seq_len)
		
		# label_token_idxs : Shape (batch_size, num_labels, label_seq_len)
		label_embs = label_embs.view(batch_size, num_labels, -1) # Shape: (batch_size,  num_labels, embed_size)
		
		# Add another dim to input_embs to score neg inputs for each input using matrix ops
		input_embs = input_embs.unsqueeze(1) # (batch_size, 1, embed_size)
		# input_embs is broadcast along second dimension so that each input is multiplied with its label embs
		# (batch_size, num_labels, embed_size) =  (batch_size, num_labels, embed_size) x (batch_size, 1, embed_size)
		temp_prod 	= label_embs*input_embs
		# Sum along dim = 2 which is embed_size dim to compute scores
		pred_label_scores 	= torch.sum(temp_prod, dim=2) # (batch_size, num_labels)
		
		assert self.config.loss_type  == "ce", f"Only cross-entropy loss supported for distillation. Loss = {self.config.loss_type}"
		
		torch_softmax = torch.nn.Softmax(dim=-1)
		target_label_scores = torch_softmax(target_label_scores)
		
		loss = F.cross_entropy(input=pred_label_scores, target=target_label_scores)
		
		return loss
	
	def compute_loss_w_negs(self, input_embs, pos_label_embs, neg_label_embs):
		"""
		Compute loss using given embeddings for positive and negative labels
		
		:param input_embs: Input embeddings. Shape (batch_size, embed_size)
		:param pos_label_embs: Embeddings of positive labels : (batch_size, embed_size)
		:param neg_label_embs: Embeddings of negative labels : (batch_size x num_negs_per_input, label_seq_len)
		:return: loss computed using given embeddings
		"""
		pos_scores 	= torch.sum(input_embs*pos_label_embs, dim=1).unsqueeze(1) # (batch_size, 1)
		
		
		# Add another dim to input_embs to score neg inputs for each input using matrix ops
		input_embs = input_embs.unsqueeze(1) # (batch_size, 1, embed_size)
		# input_embs is broadcast along second dimension so that each input is multiplied with its negatives
		# (batch_size, num_negs, embed_size) =  (batch_size, num_negs, embed_size) x (batch_size, 1, embed_size)
		temp_prod 	= neg_label_embs*input_embs
		neg_scores 	= torch.sum(temp_prod, dim=2) # (batch_size, num_negs)
		
		if self.config.loss_type  == "ce":
			batch_size = pos_scores.shape[0]
			
			final_scores = torch.cat((pos_scores, neg_scores), dim=1) # (batch_size, 1 + num_negs)
			
			# 0th col in each row contained score for positive label
			target = torch.zeros((batch_size), dtype=torch.long, device=final_scores.device)
			
			loss = F.cross_entropy(final_scores, target, reduction="mean")
		elif self.config.loss_type == "hinge" or self.config.loss_type == "hinge_sq":
	
			# Ignore positive scores that are greater than hinge_margin
			pos_scores[pos_scores > self.config.hinge_margin] = 0
			
			# Ignore negative scores that are smaller than -hinge_margin
			neg_scores[neg_scores < -self.config.hinge_margin] = 0
			
			if self.config.loss_type == "hinge":
				pos_loss = -torch.mean(pos_scores)
				neg_loss =  torch.mean(neg_scores)
			else:
				pos_scores = self.config.hinge_margin - pos_scores
				neg_scores = self.config.hinge_margin + neg_scores
				
				pos_loss = torch.mean(pos_scores*pos_scores)
				neg_loss = torch.mean(neg_scores*neg_scores)
				
			loss = (pos_loss + neg_loss)/2
		else:
			raise NotImplementedError(f"Loss type = {self.config.loss_type} not implemented")
		
		
		return loss
	
	def compute_loss_w_in_batch_negs(self, input_embs, pos_label_embs):
		"""
		Computes loss for each input using labels for other inputs as negative
		
		:param input_embs: Input embeddings. Shape (batch_size, embed_size)
		:param pos_label_embs: Embeddings of positive labels : (batch_size, embed_size)
		:return:
		"""
	
		# Encode all positive labels in the batch.
		scores = input_embs.mm(pos_label_embs.t()) # batch_size x batch_size
		batch_size = scores.shape[0]
		
		if self.config.loss_type == "ce":
			target = torch.LongTensor(torch.arange(batch_size)).type_as(scores)
			# target = target.to(self.device) # use type_as instead
			loss = F.cross_entropy(scores, target, reduction="mean")
		elif self.config.loss_type == "hinge" or self.config.loss_type == "hinge_sq":
			"""
			Loss incurred
			if score < hinge_margin for a positive input,label pair or
			if score > -hinge_margin for a negative input,label pair.
			"""
			y_mat = (2*torch.eye(batch_size) - 1).type_as(scores) # 1 where input i and label j are positive instance and -1 where negative
			loss = self.config.hinge_margin - y_mat*scores
			loss[loss < 0] = 0. # loss = max(0, loss)
			
			if self.config.loss_type == "hinge":
				loss = torch.mean(loss)
			else: # squared hinge
				loss = torch.mean(loss*loss)
		else:
			raise NotImplementedError(f"Loss type = {self.config.loss_type} not implemented")
		
		return loss
		
	def configure_optimizers(self):
		optimizer = get_bert_optimizer(models=[self],
										type_optimization=self.config.type_optimization,
										learning_rate=self.learning_rate,
										weight_decay=self.config.weight_decay,
										optimizer_type="AdamW")
		
		len_data = self.trainer.datamodule.train_data_len
		# len_data is already adjusted taking into batch_size and grad_acc_steps into account so pass 1 for these
		scheduler = get_scheduler(optimizer=optimizer,
								   epochs=self.config.num_epochs,
								   warmup_proportion=self.config.warmup_proportion,
								   len_data=len_data,
								   batch_size=1,
								   grad_acc_steps=1	)
		
		lr_scheduler_config = {
			"scheduler": scheduler,
			"interval": "step",
			"frequency": 1,
		}
		return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}
		
	def training_step(self, train_batch, batch_idx):
		if len(train_batch) == 2:
			batch_input, batch_pos_labels = train_batch
			batch_neg_labels = None
			loss = self.forward(batch_input, batch_pos_labels, batch_neg_labels)
			return loss
		elif len(train_batch) == 3 and self.config.neg_strategy in ["distill", "top_ce_match"]:
			batch_input, batch_labels, batch_label_scores = train_batch
			loss = self.forward_w_ment_ent_distill(
				input_token_idxs=batch_input,
				label_token_idxs=batch_labels,
				target_label_scores=batch_label_scores
			)
			return loss
		
		elif len(train_batch) == 3:
			batch_input, batch_pos_labels, batch_neg_labels = train_batch
			loss = self.forward(batch_input, batch_pos_labels, batch_neg_labels)
			return loss
		else:
			raise Exception(f"Invalid number of values = {len(train_batch)} to unpack in train_batch. Expected 2 or 3")
	
	def validation_step(self, val_batch, batch_idx):
		if len(val_batch) == 2:
			batch_input, batch_pos_labels = val_batch
			batch_neg_labels = None
			loss = self.forward(batch_input, batch_pos_labels, batch_neg_labels)
			return loss
		elif len(val_batch) == 3 and self.config.neg_strategy in ["distill", "top_ce_match"]:
			batch_input, batch_labels, batch_label_scores = val_batch
			loss = self.forward_w_ment_ent_distill(
				input_token_idxs=batch_input,
				label_token_idxs=batch_labels,
				target_label_scores=batch_label_scores
			)
			return loss
		
		elif len(val_batch) == 3:
			batch_input, batch_pos_labels, batch_neg_labels = val_batch
			loss = self.forward(batch_input, batch_pos_labels, batch_neg_labels)
			return loss
		else:
			raise Exception(f"Invalid number of values = {len(val_batch)} to unpack in val_batch. Expected 2 or 3")
			
	def validation_epoch_end(self, outputs):
		super(BiEncoderWrapper, self).validation_epoch_end(outputs=outputs)
	
		eval_loss = torch.mean(torch.tensor(outputs))
		self.log("dev_loss", eval_loss, sync_dist=True, on_epoch=True, logger=True)
		self.log("train_step", self.global_step, logger=True)
		self.log("train_step_frac", float(self.global_step)/self.trainer.datamodule.train_data_len, logger=True)
	
	def on_train_start(self):
		super(BiEncoderWrapper, self).on_train_start()
		
		self.log("train_step", self.global_step, logger=True)
		self.log("train_step_frac", float(self.global_step)/self.trainer.datamodule.train_data_len, logger=True)
	
	def on_train_epoch_start(self):
		super(BiEncoderWrapper, self).on_train_epoch_start()
		if self.config.reload_dataloaders_every_n_epochs and (self.current_epoch % self.config.reload_dataloaders_every_n_epochs == 0):
			LOGGER.info(f"\n\n\t\tResetting model checkpoint callback params in epoch = {self.current_epoch}\n\n")
			for checkpoint_callback in self.trainer.checkpoint_callbacks:
				checkpoint_callback.current_score = None
				checkpoint_callback.best_k_models = {}
				checkpoint_callback.kth_best_model_path = ""
				checkpoint_callback.best_model_score = None
				checkpoint_callback.best_model_path = ""
				checkpoint_callback.last_model_path = ""
			
	def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
		
		super(BiEncoderWrapper, self).on_train_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx, unused=unused)
		
		if (self.global_step + 1) % (self.config.print_interval * self.config.grad_acc_steps) == 0:
			self.log("train_loss", outputs, on_epoch=True, logger=True)
			self.log("train_step", self.global_step, logger=True)
			self.log("train_step_frac", float(self.global_step)/self.trainer.datamodule.train_data_len, logger=True)
