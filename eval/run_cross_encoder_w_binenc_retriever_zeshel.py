import os
import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import wandb
import numpy as np
from IPython import embed
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

import faiss
from utils.data_process import load_entities, load_mentions, get_context_representation
from eval.eval_utils import score_topk_preds, compute_label_embeddings
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH, MAX_PAIR_LENGTH


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def _get_cross_enc_pred(crossencoder, max_ment_length, max_pair_length, batch_ment_tokens, complete_entity_tokens_list, batch_retrieved_indices, use_all_layers):
		
	try:
		# Create pair of input,nnbr entity tokens. Strip off first token from nnbr entity as that is CLS token and also limit to max_pair_length
		if torch.is_tensor(batch_retrieved_indices):
			batch_retrieved_indices = batch_retrieved_indices.cpu().data.numpy()
		batch_nnbr_ent_tokens = [complete_entity_tokens_list[nnbr_indices].unsqueeze(0) for nnbr_indices in batch_retrieved_indices]
		batch_nnbr_ent_tokens = torch.cat(batch_nnbr_ent_tokens).to(batch_ment_tokens.device)
		
		batch_paired_inputs = []
		for i, ment_tokens in enumerate(batch_ment_tokens):
			paired_inputs = torch.stack([torch.cat((ment_tokens.view(-1), nnbr[1:]))[:max_pair_length] for nnbr in batch_nnbr_ent_tokens[i]])
			batch_paired_inputs += [paired_inputs]

		batch_paired_inputs = torch.stack(batch_paired_inputs).to(crossencoder.device)

		
		if use_all_layers:
			batch_crossenc_scores = crossencoder.score_candidate_per_layer(
				input_pair_idxs=batch_paired_inputs,
				first_segment_end=max_ment_length
			)
		else:
			
			batch_size, num_pairs, seq_len = batch_paired_inputs.shape
			batch_paired_inputs = batch_paired_inputs.reshape(batch_size*num_pairs, seq_len)
			
			pair_dataloader = DataLoader(TensorDataset(batch_paired_inputs), batch_size=500, shuffle=False)
			
			all_scores = []
			for (curr_batch,) in pair_dataloader:
				curr_batch_crossenc_scores = crossencoder.score_candidate(
					input_pair_idxs=curr_batch,
					first_segment_end=max_ment_length
				)
				all_scores += [curr_batch_crossenc_scores]
			batch_crossenc_scores = torch.cat(all_scores)
			batch_crossenc_scores = batch_crossenc_scores.reshape(batch_size, num_pairs)
			
			
		return batch_crossenc_scores
	except Exception as e:
		embed()
		raise e


def run(biencoder, crossencoder, data_fname, n_ment_start, n_ment, batch_size, top_k, res_dir, dataset_name, misc, arg_dict, use_all_layers, run_rnr_opt):
	
	if biencoder.device == torch.device("cpu"):
		wandb.alert(title="No GPUs found", text=f"{biencoder.device}, {crossencoder.device}")
		raise Exception("No GPUs found!!!")
	
	try:
		assert top_k > 1
		rng = np.random.default_rng(seed=0)
		biencoder.eval()
		crossencoder.eval()
		
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = crossencoder.tokenizer
		
		
		test_data = load_mentions(mention_file=data_fname["ment_file"], kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[n_ment_start:n_ment_start+n_ment] if n_ment > 0 else test_data
		# First extract all mentions and tokenize them
		mention_tokens_list = [get_context_representation(sample=mention, tokenizer=tokenizer, max_seq_length=MAX_MENT_LENGTH)["ids"]
								for mention in tqdm(test_data)]
		
		curr_mentions_tensor = torch.LongTensor(mention_tokens_list)
		curr_gt_labels = np.array([x["label_id"] for x in test_data])
		
		batched_data = TensorDataset(curr_mentions_tensor, torch.LongTensor(curr_gt_labels))
		bienc_dataloader = DataLoader(batched_data, batch_size=batch_size, shuffle=False)
		
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		
		label_encodings = compute_label_embeddings(
			biencoder=biencoder,
			labels_tokens_list=complete_entity_tokens_list,
			batch_size=batch_size
		)
		# label_encodings = torch.Tensor(np.load(data_fname["ent_embed_file"]))
		
		d = label_encodings.shape[-1]
		LOGGER.info(f"Building index over embeddings of shape {label_encodings.shape}")
		index = faiss.IndexFlatIP(d)
		index.add(label_encodings.cpu().numpy())
		
		label_encodings = label_encodings.t() # Take transpose for easier matrix multiplication ops later

		if run_rnr_opt:
			bienc_topk_preds = []
	
			crossenc_topk_preds_w_bienc_retrvr = defaultdict(list) if use_all_layers else []
			with torch.no_grad():
	
				LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment}, top_k={top_k}")
				LOGGER.info(f"Bi encoder model device {biencoder.device}")
				LOGGER.info(f"Cross encoder model device {crossencoder.device}")
				for batch_idx, (batch_ment_tokens, batch_ment_gt_labels) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
					batch_ment_tokens =  batch_ment_tokens.to(biencoder.device)
					curr_batch_size = batch_ment_tokens.shape[0]
	
					ment_encodings = biencoder.encode_input(batch_ment_tokens).cpu()
	
	
	
					batch_bienc_top_k_scores, batch_bienc_top_k_indices = index.search(ment_encodings.numpy(), k=top_k)
	
	
	
					_get_cross_enc_pred_w_retrvr = lambda retrv_indices : _get_cross_enc_pred(crossencoder=crossencoder,
																							  max_pair_length=MAX_PAIR_LENGTH,
																							  max_ment_length=MAX_MENT_LENGTH,
																							  batch_ment_tokens=batch_ment_tokens,
																							  complete_entity_tokens_list=complete_entity_tokens_list,
																							  batch_retrieved_indices=retrv_indices,
																							  use_all_layers=use_all_layers)
	
					# Compute cross-encoder scores for bi-encoder entities
					batch_crossenc_topk_scores_w_bienc_retrvr = _get_cross_enc_pred_w_retrvr(batch_bienc_top_k_indices)
					batch_crossenc_topk_scores_w_bienc_retrvr = batch_crossenc_topk_scores_w_bienc_retrvr.cpu().data.numpy()
	
	
					bienc_topk_preds += [(batch_bienc_top_k_indices, batch_bienc_top_k_scores)]
					if use_all_layers:
						for layer_iter in range(batch_crossenc_topk_scores_w_bienc_retrvr.shape[-1]):
							crossenc_topk_preds_w_bienc_retrvr[layer_iter] += [(batch_bienc_top_k_indices, batch_crossenc_topk_scores_w_bienc_retrvr[:,:,layer_iter])]
					else:
						crossenc_topk_preds_w_bienc_retrvr += [(batch_bienc_top_k_indices, batch_crossenc_topk_scores_w_bienc_retrvr)]
	
					wandb.log({"batch_idx": batch_idx,
							   "frac_done": float(batch_idx)/len(bienc_dataloader)})
	
	
			curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
			Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
	
			bienc_topk_preds = _get_indices_scores(bienc_topk_preds)
			if use_all_layers:
				for layer_iter in crossenc_topk_preds_w_bienc_retrvr:
					crossenc_topk_preds_w_bienc_retrvr[layer_iter] = _get_indices_scores(crossenc_topk_preds_w_bienc_retrvr[layer_iter])
			else:
				crossenc_topk_preds_w_bienc_retrvr = _get_indices_scores(crossenc_topk_preds_w_bienc_retrvr)
	
	
			json.dump(curr_gt_labels.tolist(), open(f"{curr_res_dir}/gt_labels.txt", "w"))
			json.dump(bienc_topk_preds, open(f"{curr_res_dir}/bienc_topk_preds.txt", "w"))
			json.dump(crossenc_topk_preds_w_bienc_retrvr, open(f"{curr_res_dir}/crossenc_topk_preds_w_bienc_retrvr.txt", "w"))
	
			res = {
				"bienc": score_topk_preds(
					gt_labels=curr_gt_labels,
					topk_preds=bienc_topk_preds
				),
			}
	
			if use_all_layers:
				for layer_iter in sorted(crossenc_topk_preds_w_bienc_retrvr):
					res[f"crossenc_w_bienc_retrvr_layer_{layer_iter}"] = score_topk_preds(
						gt_labels=curr_gt_labels,
						topk_preds=crossenc_topk_preds_w_bienc_retrvr[layer_iter]
					)
			else:
				res["crossenc_w_bienc_retrvr"] = score_topk_preds(
					gt_labels=curr_gt_labels,
					topk_preds=crossenc_topk_preds_w_bienc_retrvr
				)
	
			res["extra_info"] = arg_dict
	
			with open(f"{curr_res_dir}/res.json", "w") as fout:
				json.dump(res, fout, indent=4)
				LOGGER.info(json.dumps(res, indent=4))


		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise  e


def run_w_precomp_results(n_ment, batch_size, top_k, res_dir, dataset_name, misc, bi_model_file, use_all_layers, arg_dict):
	try:
		assert top_k > 1
		
		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
	
		with open(f"{curr_res_dir}/gt_labels.txt", "r") as fin:
			curr_gt_labels = json.load(fin)
		
		with open(f"{curr_res_dir}/bienc_topk_preds.txt", "r") as fin:
			bienc_topk_preds = json.load(fin)
		
		with open(f"{curr_res_dir}/crossenc_topk_preds_w_bienc_retrvr.txt", "r") as fin:
			crossenc_topk_preds_w_bienc_retrvr = json.load(fin)
		
		
		res = {
			"bienc": score_topk_preds(
				gt_labels=curr_gt_labels,
				topk_preds=bienc_topk_preds
			),
		}
		
		if use_all_layers:
			for layer_iter in sorted(crossenc_topk_preds_w_bienc_retrvr):
				res[f"crossenc_w_bienc_retrvr_layer_{layer_iter}"] = score_topk_preds(
					gt_labels=curr_gt_labels,
					topk_preds=crossenc_topk_preds_w_bienc_retrvr[layer_iter]
				)
		else:
			res["crossenc_w_bienc_retrvr"] = score_topk_preds(
				gt_labels=curr_gt_labels,
				topk_preds=crossenc_topk_preds_w_bienc_retrvr
			)
		
		res["extra_info"] = arg_dict
		
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)} for params "
					f"n_ment, batch_size, top_k, res_dir, dataset_name, misc, bi_model_file ="
					f"{n_ment}, {batch_size}, {top_k}, {res_dir}, {dataset_name}, {misc}, {bi_model_file}")
		embed()
		

def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	
	if len(indices) > 0 and torch.is_tensor(indices[0]):
		indices, scores = torch.cat(indices), torch.cat(scores)
		indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	else:
		indices, scores = np.concatenate(indices).tolist(), np.concatenate(scores).tolist()
		
	return {"indices":indices, "scores":scores}


def main():
	
	worlds = get_zeshel_world_info()
	

	parser = argparse.ArgumentParser( description='Run cross-encoder model after retrieving using biencoder model')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment_start", type=int, default=0, help="Star offset for mentions")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	
	parser.add_argument("--bi_model_file", type=str, required=True, help="Biencoder Model config file or checkpoint file")
	parser.add_argument("--use_all_layers", type=int, default=0, help="Compute score using all cross-encoder layers")
	parser.add_argument("--cross_model_file", type=str, default="", help="Crossencoder Model config file or checkpoint file")
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")
	parser.add_argument("--eval_w_precomp_preds", type=int, default=0, help="1 if files w/ predictions already exist and we only need to compute accuracy/recall etc,"
																			"0 if we also need to compute predictions")
	parser.add_argument("--run_rnr_opt", type=int, default=1, help="1 to run retrieve and rerank inference 0 otherwise")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	
	args = parser.parse_args()

	data_name = args.data_name
	n_ment = args.n_ment
	n_ment_start = args.n_ment_start
	top_k = args.top_k
	batch_size = args.batch_size
	use_all_layers = bool(args.use_all_layers)
	
	bi_model_file = args.bi_model_file
	cross_model_file = args.cross_model_file
	
	data_dir = args.data_dir
	res_dir = args.res_dir
	eval_w_precomp_preds = bool(args.eval_w_precomp_preds)
	run_rnr_opt = bool(args.run_rnr_opt)
	misc = args.misc
	disable_wandb = bool(args.disable_wandb)
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	if bi_model_file.endswith(".json"):
		with open(bi_model_file, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
	else:
		biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	
	if cross_model_file.endswith(".json"):
		with open(cross_model_file, "r") as fin:
			config = json.load(fin)
			if use_all_layers:
				config["bert_args"] = {"output_attentions":True, "output_hidden_states":True}
			crossencoder = CrossEncoderWrapper.load_model(config=config)
	else:
		crossencoder = CrossEncoderWrapper.load_from_checkpoint(cross_model_file)
		
		
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	iter_worlds = worlds if data_name == "all" else [("dummy", data_name)]
	
	config = {
			"goal": "Compute entity-entity pairwise similarity matrix",
			"batch_size":batch_size,
			
			"CUDA_DEVICE":os.environ["CUDA_VISIBLE_DEVICES"]
		}
	config.update(args.__dict__)
	
	try:
		wandb.init(
			project="CrossEnc-Rerank-w-Bienc-Retr",
			dir=res_dir,
			config=config,
			mode="disabled" if disable_wandb else "online"
		)
	except:
		try:
			wandb.init(
				project="CrossEnc-Rerank-w-Bienc-Retr",
				dir=res_dir,
				config=config,
				settings=wandb.Settings(start_method="fork"),
				mode="disabled" if disable_wandb else "online"
			)
		
		except Exception as e:
			LOGGER.info(f"Error raised = {e}")
			LOGGER.info("Running wandb in offline mode")
			wandb.init(
				project="CrossEnc-Rerank-w-Bienc-Retr",
				dir=res_dir,
				config=config,
				mode="disabled" if disable_wandb else "offline"
			)
	
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		if not eval_w_precomp_preds:
			run(
				biencoder=biencoder,
				crossencoder=crossencoder,
				data_fname=DATASETS[world_name],
				n_ment_start=n_ment_start,
				n_ment=n_ment,
				top_k=top_k,
				batch_size=batch_size,
				dataset_name=world_name,
				res_dir=res_dir,
				misc=misc,
				arg_dict=args.__dict__,
				use_all_layers=use_all_layers,
				run_rnr_opt=run_rnr_opt
			)
		else:
			run_w_precomp_results(
				n_ment=n_ment,
				top_k=top_k,
				batch_size=batch_size,
				dataset_name=world_name,
				res_dir=res_dir,
				misc=misc,
				bi_model_file=bi_model_file,
				use_all_layers=use_all_layers,
				arg_dict=args.__dict__
			)
			


if __name__ == "__main__":
	main()

