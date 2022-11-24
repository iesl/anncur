import os
import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path
import pickle

import wandb
from torch.utils.data import DataLoader, TensorDataset

from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH
from utils.data_process import load_entities, load_mentions, get_context_representation
from models.crossencoder import CrossEncoderWrapper


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)




def create_paired_dataset(encoded_mentions, encoded_entities, max_pair_length, batch_size, num_pairs_per_input):
	"""
	Create a list of representations by pairing each mention with each entity
	Dataset consists of list of batches,
	 each batch consists batch_size groups of pairs, and
	 each group consists of num_pairs_per_input ment-entity pairs
	:param encoded_mentions:
	:param encoded_entities:
	:param max_pair_length:
	:param batch_size
	:param num_pairs_per_input
	:return:
	"""
	try:
		paired_dataset = []
	
		def get_pairs():
			for mention in encoded_mentions:
				for ent in encoded_entities:
					pair = np.concatenate((mention, ent[1:]))
					yield pair[:max_pair_length]
	
		curr_batch = []
		for pair in get_pairs():
			curr_batch += [pair]
			if len(curr_batch) == num_pairs_per_input:
				paired_dataset += [curr_batch]
				curr_batch = []
	
		if len(curr_batch) > 0:
			# padded_curr_batch = np.zeros( (num_pairs_per_input, len(curr_batch[0])), dtype=int) # Use all zeros array to pad the current batch
			padded_curr_batch = np.array([curr_batch[-1]]*num_pairs_per_input) # Repeat last element to pad the current batch
			padded_curr_batch[:len(curr_batch), :] = curr_batch
			paired_dataset += [padded_curr_batch]
	
		paired_dataset = torch.LongTensor(paired_dataset)
	
		tensor_data = TensorDataset(paired_dataset)
		dataloader = DataLoader(
			tensor_data, batch_size=batch_size, shuffle=False
		)
		return dataloader
	except Exception as e:
		embed()
		raise e


def _run_cross_encoder(cross_encoder, dataloader, max_ment_length, n_ment, n_ent, use_all_layers, log=True):
	"""
	Run cross-encoder model on given data and return scores
	:param cross_encoder:
	:param dataloader:
	:param max_ment_length:
	:param use_all_layers: Compute scores using all_layers of cross-encoder model
	:return:
	"""
	try:
		all_scores_list = []
		all_scores_per_layer_list = []
		for step, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
			batch_input, = batch
			batch_input = batch_input.to(cross_encoder.device)
			
			if use_all_layers:
				batch_score_per_layer = cross_encoder.score_candidate_per_layer(batch_input, first_segment_end=max_ment_length)
				all_scores_per_layer_list += [batch_score_per_layer.squeeze(0)]
			else:
				batch_score = cross_encoder.score_candidate(batch_input, first_segment_end=max_ment_length)
				all_scores_list += [batch_score.squeeze(0)]
			
			if log:
				wandb.log({"batch_idx": step,
						   "frac_done": float(step)/len(dataloader)})
		
		if use_all_layers:
			all_scores_per_layer = torch.cat(all_scores_per_layer_list, dim=0) # shape:
			all_scores_per_layer = all_scores_per_layer[:n_ment*n_ent] # Remove scores for padded data to get shape = n_ment*n_ent x num_layers
			# assert all_scores_per_layer.shape ==  n_ment*n_ent, num_layers, f"SHape of all_scores_per_layer = {all_scores_per_layer.shape} !=  {n_ment*n_ent, num_layers}"
			
			all_scores_per_layer = all_scores_per_layer.transpose(0, 1) # Shape = num_layers x n_ment*n_ent
			all_scores_per_layer = all_scores_per_layer.view(-1, n_ment, n_ent).cpu() # Shape : num_layers x n_ment x n_ent
			return all_scores_per_layer
		else:
			all_scores = torch.cat(all_scores_list)
			all_scores = all_scores[:n_ment*n_ent] # Remove scores for padded data to get shape = (n_ment*n_ent,)
			all_scores = all_scores.view(n_ment, n_ent).cpu() # shape: n_ment x n_ent
			return all_scores
		
	except Exception as e:
		embed()
		raise e


def _run_cross_encoder_for_embeds(cross_encoder, dataloader, max_ment_length, n_ment, n_ent, log=True):
	"""
	Run cross-encoder model on given data and return contextualized embeddings of inputs and labels
	:param cross_encoder:
	:param dataloader:
	:param max_ment_length:
	
	:return:
	"""
	try:
		all_input_embeds, all_label_embeds = [], []
		for step, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
			batch_input, = batch
			batch_input = batch_input.to(cross_encoder.device)
			
			curr_input_embeds, curr_label_embeds = cross_encoder.embed_paired_input_and_labels(batch_input, first_segment_end=max_ment_length)
			all_input_embeds += [curr_input_embeds.cpu()]
			all_label_embeds += [curr_label_embeds.cpu()]
			
			if log: wandb.log({"batch_idx": step, "frac_done": float(step)/len(dataloader)})
		
		all_input_embeds = torch.cat(all_input_embeds) # Shape: (batch_size*num_batches, embed_dim)
		assert len(all_input_embeds.shape) == 2, f"len(all_input_embeds.shape) = {len(all_input_embeds.shape)} != 2"
		embed_dim =  all_input_embeds.shape[1]
		all_input_embeds = all_input_embeds[:n_ment*n_ent] # Remove embeds for padded data to get shape = (n_ment*n_ent, embed_dim)
		all_input_embeds = all_input_embeds.view(n_ment, n_ent, embed_dim) # Shape: (n_ment, n_ent, embed_dim)
		
		all_label_embeds = torch.cat(all_label_embeds) # Shape: (batch_size*num_batches, embed_dim)
		assert len(all_label_embeds.shape) == 2, f"len(all_label_embeds.shape) = {len(all_label_embeds.shape)} != 2"
		embed_dim =  all_label_embeds.shape[1]
		all_label_embeds = all_label_embeds[:n_ment*n_ent] # Remove embeds for padded data to get shape = (n_ment*n_ent, embed_dim)
		all_label_embeds = all_label_embeds.view(n_ment, n_ent, embed_dim) # Shape: (n_ment, n_ent, embed_dim)
		
		return all_input_embeds, all_label_embeds
	
	except Exception as e:
		embed()
		raise e


def run(crossencoder, data_fname, n_ment_start, n_ment, n_ent, mode,  batch_size, dataset_name, res_dir, use_all_layers, misc, arg_dict):
	
	if crossencoder.device == torch.device("cpu"):
		wandb.alert(title="No GPUs found", text=f"{crossencoder.device}")
		raise Exception("No GPUs found!!!")
	try:
		
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = crossencoder.tokenizer
		
		LOGGER.info("Loading test samples")
		test_data = load_mentions(mention_file=data_fname["ment_file"],
								  kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[n_ment_start:n_ment_start+n_ment] if n_ment > 0 else test_data
		
		LOGGER.info(f"Tokenize {n_ment} test samples")
		# First extract all mentions and tokenize them
		mention_tokens_list = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=MAX_MENT_LENGTH)["ids"]
								for mention in tqdm(test_data)]
		
		
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		# complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list).to(biencoder.device)
		
		n_ent = len(complete_entity_tokens_list) if n_ent < 0 else n_ent
		LOGGER.info(f"Running score computation with first {n_ent} entities!!!")
		entity_id_list = np.arange(n_ent)
		entity_tokens_list = complete_entity_tokens_list[entity_id_list]
	
		with torch.no_grad():
			LOGGER.info(f"Computing score for each test entity with each mention. batch_size={batch_size}, n_ment={n_ment}, n_ent={n_ent}")
			dataloader = create_paired_dataset(
				encoded_mentions=mention_tokens_list,
				encoded_entities=entity_tokens_list,
				max_pair_length=MAX_PAIR_LENGTH,
				batch_size=1,
				num_pairs_per_input=batch_size
			)
			
			LOGGER.info("Running cross encoder model now")
			crossencoder.eval()
			
			if mode == "scores":
				ment_to_ent_scores = _run_cross_encoder(
					cross_encoder=crossencoder,
					dataloader=dataloader,
					max_ment_length=MAX_MENT_LENGTH,
					n_ment=n_ment,
					n_ent=n_ent,
					use_all_layers=use_all_layers,
					log=True
				)
				LOGGER.info(f"Computed score matrix of shape = {ment_to_ent_scores.shape}")
				
				LOGGER.info(f"ment_to_ent_scores :\n{ment_to_ent_scores}")
				curr_res_dir = f"{res_dir}/{dataset_name}"
				Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
				with open(f"{curr_res_dir}/ment_to_ent_scores_n_m_{n_ment}_n_e_{n_ent}_all_layers_{use_all_layers}{misc}.pkl", "wb") as fout:
					res = {
						"ment_to_ent_scores":ment_to_ent_scores,
						"ment_to_ent_scores.shape":ment_to_ent_scores.shape,
						"test_data":test_data,
						"mention_tokens_list":mention_tokens_list,
						"entity_id_list":entity_id_list,
   						"entity_tokens_list":entity_tokens_list,
						"arg_dict":arg_dict
					}
					pickle.dump(res, fout)
			elif mode == "embeds":
				all_input_embeds, all_label_embeds = _run_cross_encoder_for_embeds(
					cross_encoder=crossencoder,
					dataloader=dataloader,
					max_ment_length=MAX_MENT_LENGTH,
					n_ment=n_ment,
					n_ent=n_ent,
					log=True
				)
				
				LOGGER.info(f"Computed embedding matrix of shape = {all_input_embeds.shape} and {all_label_embeds.shape}")
				
				# ment_to_ent_scores = torch.sum(all_input_embeds*all_label_embeds, dim=-1)
				# LOGGER.info(f"ment_to_ent_scores :\n{ment_to_ent_scores}")
				
				curr_res_dir = f"{res_dir}/{dataset_name}"
				Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
				with open(f"{curr_res_dir}/ment_and_ent_embeds_n_m_{n_ment}_n_e_{n_ent}_all_layers_{use_all_layers}{misc}.pkl", "wb") as fout:
					res = {
						"all_label_embeds":all_label_embeds,
						"all_input_embeds":all_input_embeds,
						"all_label_embeds.shape":all_label_embeds.shape,
						"all_input_embeds.shape":all_input_embeds.shape,
						"test_data":test_data,
						"mention_tokens_list":mention_tokens_list,
						"entity_id_list":entity_id_list,
						"entity_tokens_list":entity_tokens_list,
						"arg_dict":arg_dict
					}
					pickle.dump(res, fout)
					
			else:
				raise NotImplementedError(f"Mode = {mode} not supported")
			
			
		wandb.log({"frac_done": 1.1})
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def main():
	exp_id = "4_Zeshel_Ment2Ent"
	data_dir = "../../data/zeshel"
	
	
	worlds = get_zeshel_world_info()
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	parser = argparse.ArgumentParser( description='Run cross-encoder model for computing mention-entity scoring matrix')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment_start", type=int, default=0, help="Start offset for mention indexing")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--n_ent", type=int, default=-1, help="Number of entities.  -1 for all entities")
	parser.add_argument("--mode", type=str, choices=["scores", "embeds"], default="scores", help="Type of computation to do - whether to compute scores or embedding of inputs and labels")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	
	parser.add_argument("--res_dir", type=str, required=True, help="Directory to save results")
	parser.add_argument("--cross_model_config", type=str, default="", help="Crossencoder Model config file")
	parser.add_argument("--cross_model_ckpt", type=str, default="", help="Crossencoder Model ckpt file from pytorch lightning")
	parser.add_argument("--layers", type=str, choices=["final", "all"], help="Choose whether to compute scores for each layer of crossencoder or just final layer")
	parser.add_argument("--misc", type=str, default="", help="misc suffix for output file")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0,1], help="Disable wandb")
	
	
	args = parser.parse_args()

	data_name = args.data_name
	n_ment_start = args.n_ment_start
	n_ment = args.n_ment
	n_ent = args.n_ent
	mode = args.mode
	disable_wandb = args.disable_wandb
	batch_size = args.batch_size
	
	res_dir = args.res_dir
	cross_model_config = args.cross_model_config
	cross_model_ckpt = args.cross_model_ckpt
	assert cross_model_config != "" or cross_model_ckpt != "" , "Both cross_model_config and cross_model_ckpt can't be empty"
	assert cross_model_config == "" or cross_model_ckpt == "" , "Both cross_model_config and cross_model_ckpt can't be non-empty"
	
	layers = args.layers
	assert layers == "all" or layers == "final"
	use_all_layers = layers == "all"
	misc = args.misc
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	if cross_model_config != "":
		with open(cross_model_config, "r") as fin:
			config = json.load(fin)
			if use_all_layers:
				config["bert_args"] = {"output_attentions":True, "output_hidden_states":True}
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_ckpt)
		
	config = {
		"goal": "Compute n_m x n_e pairwise similarity matrix",
		"n_ment":n_ment,
		"batch_size":batch_size,
		"n_ent":n_ent,
		"data_name":data_name,
		"layers":layers,
		"misc":misc,
   		"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
	}
	config.update(args.__dict__)
	
	try:
		wandb.init(
			project=exp_id,
			dir=res_dir,
			config=config,
			mode="disabled" if disable_wandb else "online"
		)
	
	except Exception as e:
		try:
			LOGGER.info(f"Trying with wandb.Settings(start_method=fork) as error = {e} as raised")
			wandb.init(
				project=exp_id,
				dir=res_dir,
				config=config,
				mode="disabled" if disable_wandb else "online",
				settings=wandb.Settings(start_method="fork")
			)
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(
				project=exp_id,
				dir=res_dir,
				config=config,
				mode="offline"
			)
	
	iter_worlds = worlds if data_name == "all" else [("", data_name)]
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		run(
			crossencoder=crossencoder,
			data_fname=DATASETS[world_name],
			dataset_name=data_name,
			n_ment_start=n_ment_start,
			n_ment=n_ment,
			n_ent=n_ent,
			batch_size=batch_size,
			res_dir=res_dir,
			use_all_layers=use_all_layers,
			misc=misc,
			arg_dict=args.__dict__,
			mode=mode
		)
	


if __name__ == "__main__":
	main()
