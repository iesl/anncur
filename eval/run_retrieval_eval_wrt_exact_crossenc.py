import os
import sys
import json
import torch
import pickle
import logging
import argparse
import wandb
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from eval.eval_utils import compute_label_embeddings, compute_input_embeddings, compute_overlap
from eval.matrix_approx_zeshel import CURApprox, plot_heat_map
from models.biencoder import BiEncoderWrapper
from models.crossencoder import CrossEncoderWrapper
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def _get_indices_scores(topk_preds):
	"""
	Convert a list of indices,scores tuple to two list by concatenating all indices and all scores together.
	:param topk_preds: List of indices,scores tuple
	:return: dict with two keys "indices" and "scores" mapping to lists
	"""
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy(), scores.cpu().numpy()
	return {"indices":indices, "scores":scores}


def run_approx_eval_w_seed(approx_method, all_ment_to_ent_scores, n_ment_anchors, n_ent_anchors, top_k, top_k_retvr, seed, precomp_approx_ment_to_ent_scores):
	"""
	Takes a ground-truth mention x entity matrix as input, uses some rows and columns of this matrix to approximate the entire matrix
	and also evaluate the approximation
	:param all_ment_to_ent_scores:
	:param precomp_approx_ment_to_ent_scores
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param top_k_retvr:
	:param approx_method
	:param seed:
	:return:
	"""

	
	try:
		n_ments = all_ment_to_ent_scores.shape[0]
		n_ents = all_ment_to_ent_scores.shape[1]
	
		rng = np.random.default_rng(seed=seed)
	
		anchor_ment_idxs = sorted(rng.choice(n_ments, size=n_ment_anchors, replace=False))
		anchor_ent_idxs = sorted(rng.choice(n_ents, size=n_ent_anchors, replace=False))
		row_idxs = anchor_ment_idxs
		col_idxs = anchor_ent_idxs
		rows = all_ment_to_ent_scores[row_idxs,:]
		cols = all_ment_to_ent_scores[:,col_idxs]
	
		non_anchor_ment_idxs = sorted(list(set(list(range(n_ments))) - set(anchor_ment_idxs)))
		non_anchor_ent_idxs = sorted(list(set(list(range(n_ents))) - set(anchor_ent_idxs)))
	
		if approx_method in ["bienc","fixed_anc_ent"] or approx_method.startswith("fixed_anc_ent_cur_"):
			approx_ment_to_ent_scores = precomp_approx_ment_to_ent_scores
		elif approx_method == "cur":
			# approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows", A = all_ment_to_ent_scores)
			approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows")
			approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
		elif approx_method == "cur_oracle":
			# approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows", A = all_ment_to_ent_scores)
			approx = CURApprox(row_idxs=row_idxs, col_idxs=col_idxs, rows=rows, cols=cols, approx_preference="rows", A=all_ment_to_ent_scores)
			approx_ment_to_ent_scores = approx.get(list(range(n_ments)), list(range(n_ents)))
		else:
			raise NotImplementedError(f"approx_method = {approx_method} not supported")
		
		topk_preds = []
		approx_topk_preds = []
		topk_w_approx_retrvr_preds = []
		
	
		for ment_idx in range(n_ments):
	
			curr_ment_scores = all_ment_to_ent_scores[ment_idx]
			approx_curr_ment_scores = approx_ment_to_ent_scores[ment_idx]
	
			# Get top-k indices from exact matrix
			top_k_scores, top_k_indices = curr_ment_scores.topk(top_k)
	
			# Get top-k indices from approx-matrix
			approx_top_k_scores, approx_top_k_indices = approx_curr_ment_scores.topk(top_k_retvr)
			
			# Re-rank top-k indices from approx-matrix using exact scores from ment_to_ent matrix
			# Scores from given ment_to_ent matrix filled only for entities retrieved by approximate retriever
			temp = torch.zeros(curr_ment_scores.shape) - 99999999999999
			temp[approx_top_k_indices] = curr_ment_scores[approx_top_k_indices]
		
			top_k_w_approx_retrvr_scores, top_k_w_approx_retrvr_indices = temp.topk(top_k)
			
			topk_preds += [(top_k_indices.unsqueeze(0), top_k_scores.unsqueeze(0))]
			approx_topk_preds += [(approx_top_k_indices.unsqueeze(0), approx_top_k_scores.unsqueeze(0))]
			topk_w_approx_retrvr_preds += [(top_k_w_approx_retrvr_indices.unsqueeze(0), top_k_w_approx_retrvr_scores.unsqueeze(0))]
	
		
		topk_preds = _get_indices_scores(topk_preds)
		approx_topk_preds = _get_indices_scores(approx_topk_preds)
		topk_w_approx_retrvr_preds = _get_indices_scores(topk_w_approx_retrvr_preds)
		
		def score_topk_preds_wrapper(arg_ment_idxs):
			
			exact_vs_reranked_approx_retvr = compute_overlap(
				indices_list1=topk_preds["indices"][arg_ment_idxs],
				indices_list2=topk_w_approx_retrvr_preds["indices"][arg_ment_idxs],
			)
			new_exact_vs_reranked_approx_retvr = {}
			for _metric in exact_vs_reranked_approx_retvr:
				new_exact_vs_reranked_approx_retvr[f"{_metric}_mean"] = float(exact_vs_reranked_approx_retvr[_metric][0][5:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_std"] = float(exact_vs_reranked_approx_retvr[_metric][1][4:])
				new_exact_vs_reranked_approx_retvr[f"{_metric}_p50"] = float(exact_vs_reranked_approx_retvr[_metric][2][4:])
			
		
			res = {
				f"exact_vs_reranked_approx_retvr" : new_exact_vs_reranked_approx_retvr,
			}
			
			new_res = {}
			for res_type in res:
				for metric in res[res_type]:
					new_res[f"{res_type}~{metric}"] = res[res_type][metric]
			
			new_res["approx_error"] =  (torch.norm( (approx_ment_to_ent_scores - all_ment_to_ent_scores)[arg_ment_idxs, :] )).data.numpy()
			new_res["approx_error_relative"] =  new_res["approx_error"]/ (torch.norm(all_ment_to_ent_scores[arg_ment_idxs, :]).data.numpy())
			return new_res
		
		final_res = {
			"anchor":score_topk_preds_wrapper(arg_ment_idxs=anchor_ment_idxs),
			"non_anchor":score_topk_preds_wrapper(arg_ment_idxs=non_anchor_ment_idxs),
			"all":score_topk_preds_wrapper(arg_ment_idxs=list(range(n_ments)))
		}
		return final_res
	except Exception as e:
		embed()
		raise e



def run_approx_eval(approx_method, all_ment_to_ent_scores, precomp_approx_ment_to_ent_scores, n_ment_anchors, n_ent_anchors, top_k, top_k_retvr, n_seeds):
	"""
	Run approximation eval for different seeds
	:param approx_method
	:param all_ment_to_ent_scores:
	:param precomp_approx_ment_to_ent_scores
	:param n_ment_anchors:
	:param n_ent_anchors:
	:param top_k:
	:param top_k_retvr
	:param n_seeds:
	:return:
	"""
	
	avg_res = defaultdict(lambda : defaultdict(list))
	for seed in range(n_seeds):
		res = run_approx_eval_w_seed(
			approx_method=approx_method,
			all_ment_to_ent_scores=all_ment_to_ent_scores,
			precomp_approx_ment_to_ent_scores=precomp_approx_ment_to_ent_scores,
			n_ment_anchors=n_ment_anchors,
			n_ent_anchors=n_ent_anchors,
			top_k=top_k,
			top_k_retvr=top_k_retvr,
			seed=seed,
		)
		
		# Accumulate results for each seed
		for ment_type, res_dict in res.items():
			for metric, val in res_dict.items():
				avg_res[ment_type][metric] += [float(val)]
	
	# Average results for all different seeds
	new_avg_res = defaultdict(dict)
	for ment_type in avg_res:
		for metric in avg_res[ment_type]:
			new_avg_res[ment_type][metric] = np.mean(avg_res[ment_type][metric])
	
	return new_avg_res
	

def run(base_res_dir, data_info, n_seeds, batch_size, plot_only, misc, disable_wandb, arg_dict, biencoder=None):
	
	try:
		
		if biencoder: biencoder.eval()
		data_name, data_fname = data_info
	
		LOGGER.info("Loading precomputed ment_to_ent scores")
		with open(data_fname["crossenc_ment_to_ent_scores"], "rb") as fin:
			dump_dict = pickle.load(fin)
			crossenc_ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			test_data = dump_dict["test_data"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
		
		complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
		mention_tokens_list = torch.LongTensor(mention_tokens_list)
		total_n_ment, total_n_ent = crossenc_ment_to_ent_scores.shape
		
		
		
		# For plots
		eval_methods  = ["cur", "cur_oracle"]

		n_ment_anchors_vals = [50, 100, 200, 500, 1000, 2000, 5000]
		# n_ment_anchors_vals = [50, 100, 200]
		n_ment_anchors_vals = [v for v in n_ment_anchors_vals if v <= total_n_ment]
		
		n_ent_anchors_vals = [50, 100, 200, 500, 1000, 2000]
		# n_ent_anchors_vals = [50, 100, 200]
		n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v < total_n_ent] + [total_n_ent]

		top_k_vals = [1, 10, 50, 100]
		top_k_retr_vals = [100, 500, 1000]
		top_k_vals = [10]
		top_k_retr_vals = [500]
		top_k_retr_vals_bienc = top_k_retr_vals
		
		
		# For debug
		# eval_methods  = ["fixed_anc_ent", "cur", "fixed_anc_ent_cur_100", "fixed_anc_ent_cur_1000"]
		# eval_methods  = ["cur", "fixed_anc_ent", "fixed_anc_ent_cur_100", "fixed_anc_ent_cur_1000"]
		# eval_methods  = ["cur"]
		# n_ment_anchors_vals = [50, 100, 200]
		# n_ent_anchors_vals = [50, 100, 1000]
		# top_k_vals = [1, 10, 50]
		# top_k_retr_vals = [50, 500]
		# top_k_retr_vals_bienc = top_k_retr_vals
		
		res_dir = f"{base_res_dir}/nm={total_n_ment}_ne={total_n_ent}_s={n_seeds}{misc}"
		Path(res_dir).mkdir(exist_ok=True, parents=True)
		other_args = {'arg_dict': arg_dict, 'top_k_vals': top_k_vals, 'top_k_retr_vals': top_k_retr_vals,
					  'n_ent_anchors_vals': n_ent_anchors_vals, 'n_ment_anchors_vals': n_ment_anchors_vals}
			
		if not plot_only:
			wandb.init(
				project="CUR_vs_Bienc_Eval",
				dir=res_dir,
				config=other_args,
				mode="disabled" if disable_wandb else "online"
			)
			
			eval_res = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
			for curr_method in eval_methods:
	
				LOGGER.info(f"Running inference for method = {curr_method}")
				precomp_approx_ment_to_ent_scores = {}
				if curr_method  == "bienc":
					label_embeds = compute_label_embeddings(
						biencoder=biencoder,
						labels_tokens_list=complete_entity_tokens_list,
						batch_size=batch_size
					)
	
					mention_embeds = compute_input_embeddings(
						biencoder=biencoder,
						input_tokens_list=mention_tokens_list,
						batch_size=batch_size
					)
					bienc_ment_to_ent_scores = mention_embeds @ label_embeds.T
					precomp_approx_ment_to_ent_scores = {x:bienc_ment_to_ent_scores for x in n_ent_anchors_vals}
				elif curr_method == "cur":
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method == "cur_oracle":
					precomp_approx_ment_to_ent_scores = {x:None for x in n_ent_anchors_vals}
				elif curr_method == "fixed_anc_ent":
					
					ment_to_ent_scores_wrt_anc_ents = {}
					ent2ent_dir = os.path.dirname(data_fname["crossenc_ment_to_ent_scores"])
					for n_anc_ent in n_ent_anchors_vals:
						e2e_fname = f"{ent2ent_dir}/ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_topk_{n_anc_ent}_embed_bienc_m2e_bienc_cluster.pkl"
						
						if not os.path.isfile(e2e_fname):
							LOGGER.info(f"File for n_anc_ent={n_anc_ent} not found")  #"i.e. {e2e_fname} does not exist")
							continue
						
						with open(e2e_fname, "rb") as fin:
							dump_dict = pickle.load(fin)
							ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : Number of entities x Number of anchors
							anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of anchors
							
						mention_embeds = crossenc_ment_to_ent_scores[:, anchor_ents]
						
						ment_to_ent_scores_wrt_anc_ents[n_anc_ent] = mention_embeds @ ent_embeds.T
						
						
					precomp_approx_ment_to_ent_scores = ment_to_ent_scores_wrt_anc_ents
				elif curr_method.startswith("fixed_anc_ent_cur_"):
					
					n_fixed_anchor_ents = int(curr_method[len("fixed_anc_ent_cur_"):])
					
					ent2ent_dir = os.path.dirname(data_fname["crossenc_ment_to_ent_scores"])
					e2e_fname = f"{ent2ent_dir}/ent_to_ent_scores_n_e_{NUM_ENTS[data_name]}x{NUM_ENTS[data_name]}_topk_{n_fixed_anchor_ents}_embed_bienc_m2e_bienc_cluster.pkl"
					
					if not os.path.isfile(e2e_fname):
						LOGGER.info(f"File for num_fixed_ent={n_fixed_anchor_ents} not found")  #"i.e. {e2e_fname} does not exist")
						continue
					
					with open(e2e_fname, "rb") as fin:
						dump_dict = pickle.load(fin)
						ent_embeds = dump_dict["ent_to_ent_scores"] # Shape : (Number of entities, Number of anchors)
						fixed_anchor_ents = dump_dict["topk_ents"][0] # Shape : Number of fixed anchor entities
						n_ents = ent_embeds.shape[0]
						
						R = ent_embeds.T # shape : (n_fixed_anchor_ents, n_ents)
						
					rng = np.random.default_rng(seed=0)
					ment_to_ent_scores_wrt_anc_ents = {}
					for n_anc_ent in n_ent_anchors_vals:
						
						anchor_ent_idxs = sorted(rng.choice(n_ents, size=n_anc_ent, replace=False))
						
						intersect_mat = R[:, anchor_ent_idxs] # shape: (n_fixed_anchor_ents, n_anc_ent)
						U = torch.tensor(np.linalg.pinv(intersect_mat))  # shape: (n_anc_ent, n_fixed_anchor_ents)
						UR = U @ R # shape: (n_anc_ent, n_ents)
						
						# Score of mentions w/ anchor entities,
						mention_embeds = crossenc_ment_to_ent_scores[:, anchor_ent_idxs] # shape: (n_ments, n_anc_ent)
						
						# (n_ments, n_ents) = (n_ments, n_anc_ent) x (n_anc_ent, n_ents)
						ment_to_ent_scores_wrt_anc_ents[n_anc_ent] = mention_embeds @ UR
						
						
					precomp_approx_ment_to_ent_scores = ment_to_ent_scores_wrt_anc_ents
				
				else:
					raise NotImplementedError(f"Method = {curr_method} not supported")
				
				if curr_method == "bienc":
					val_tuple_list = list(itertools.product(top_k_vals, top_k_retr_vals_bienc, n_ment_anchors_vals, n_ent_anchors_vals))
				else:
					val_tuple_list = list(itertools.product(top_k_vals, top_k_retr_vals, n_ment_anchors_vals, n_ent_anchors_vals))
					
				for ctr, (top_k, top_k_retvr, n_ment_anchors, n_ent_anchors) in tqdm(enumerate(val_tuple_list), total=len(val_tuple_list)):
					wandb.log({f"ctr_{curr_method}":ctr/len(val_tuple_list)})
					if top_k_retvr < top_k: continue
					if top_k_retvr > total_n_ent: continue
					if n_ent_anchors not in precomp_approx_ment_to_ent_scores: continue
					
					if curr_method == "bienc" and n_ent_anchors != n_ent_anchors_vals[0]:
						prev_ans = eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors_vals[0]}"]
						eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = prev_ans
						continue
						
					if curr_method.startswith("bienc") and n_ment_anchors != n_ment_anchors_vals[0]:
						prev_ans = eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors_vals[0]}~anc_n_e={n_ent_anchors}"]
						eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = prev_ans
						continue
						
					curr_ans = run_approx_eval(
						approx_method=curr_method,
						all_ment_to_ent_scores=crossenc_ment_to_ent_scores,
						precomp_approx_ment_to_ent_scores=precomp_approx_ment_to_ent_scores[n_ent_anchors],
						n_ment_anchors=n_ment_anchors,
						n_ent_anchors=n_ent_anchors,
						top_k=top_k,
						top_k_retvr=top_k_retvr,
						n_seeds=n_seeds
					)
	
					eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"] = curr_ans
	
		
			res_fname = f"{res_dir}/retrieval_wrt_exact_crossenc.json"
			eval_res["other_args"] = other_args
			with open(res_fname, "w") as fout:
				json.dump(obj=eval_res, fp=fout, indent=4)


		plot(
			res_dir=res_dir,
			method_vals=eval_methods,
		)
		
	except Exception as e:
		embed()
		raise e
	



def plot(res_dir, method_vals):
	
	try:
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]
		LOGGER.info("Now plotting results")
		res_fname = f"{res_dir}/retrieval_wrt_exact_crossenc.json"
		with open(res_fname, "r") as fin:
			eval_res = json.load(fp=fin)
	
		n_ment_anchors_vals = eval_res["other_args"]["n_ment_anchors_vals"]
		n_ent_anchors_vals = eval_res["other_args"]["n_ent_anchors_vals"]
		top_k_vals = eval_res["other_args"]["top_k_vals"]
		top_k_retvr_vals = eval_res["other_args"]["top_k_retr_vals"]
		n_ent = NUM_ENTS[eval_res["other_args"]["arg_dict"]["data_name"]]
		
		############################# NOW VISUALIZE RESULTS AS A FUNCTION OF N_ANCHORS ####################################
		# metrics = [f"exact_vs_reranked_approx_retvr~{m1}_{m2}" for m1 in ["common", "diff", "total", "common_frac", "diff_frac"] for m2 in ["mean", "std", "p50"]]
		metrics = [f"exact_vs_reranked_approx_retvr~{m1}_{m2}" for m1 in ["common_frac"] for m2 in ["mean"]]
	
		# mtype_vals = ["non_anchor", "all", "anchor"]
		mtype_vals = ["non_anchor"]
		for mtype, curr_method, top_k, top_k_retvr, metric in itertools.product(mtype_vals, method_vals, top_k_vals, top_k_retvr_vals, metrics):
			if top_k > top_k_retvr: continue
			val_matrix = []
			# Build matrix for given topk value with varying number of anchor mentions and anchor entities
			try:
				for n_ment_anchors in n_ment_anchors_vals:
					# curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
					curr_config_res = [100*eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
									   if f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}" in eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"]
									   else 0.
									   for n_ent_anchors in n_ent_anchors_vals]
					val_matrix += [curr_config_res]
			except KeyError as e:
				LOGGER.info(f"Key-error = {e} for mtype = {mtype}, curr_method={curr_method}, top_k={top_k}, top_k_retvr={top_k_retvr}, metric={metric}")
				# embed()
				continue
	
			val_matrix = np.array(val_matrix, dtype=np.float64)
			curr_res_dir = f"{res_dir}/plots_{mtype}/k={top_k}/separate_plots/k_retr={top_k_retvr}_{curr_method}"
			Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
			
			# LOGGER.info(f"Saving file {curr_res_dir}")
			# embed()
			plot_heat_map(
				val_matrix=val_matrix,
				row_vals=n_ment_anchors_vals,
				col_vals=n_ent_anchors_vals,
				metric=metric,
				top_k=top_k,
				curr_res_dir=curr_res_dir
			)
	
		
		LOGGER.info("Now plotting recall-vs-cost plots for comparing bienc and cur method")
		################################################################################################################
		mtype_vals = ["non_anchor"]
		metrics = [f"exact_vs_reranked_approx_retvr~{m1}_{m2}" for m1 in ["common_frac"] for m2 in ["mean"]]
		
		for mtype, top_k, metric in itertools.product(mtype_vals, top_k_vals, metrics):
			try:
				plt.clf()
				y_vals = defaultdict(list)
				x_vals = defaultdict(list)
				
				
				for n_ment_anchors, n_ent_anchors, top_k_retvr in itertools.product(n_ment_anchors_vals, n_ent_anchors_vals, top_k_retvr_vals):
					if top_k > top_k_retvr: continue
					# curr_config_res = [eval_res[f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}~k={top_k}"][mtype][metric] for n_ent_anchors in n_ent_anchors_vals]
					y_vals["cur"] += [eval_res["cur"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]]
					y_vals["bienc"] += [eval_res["bienc"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]]
				
					x_vals["cur"] += [top_k_retvr + n_ent_anchors]
					x_vals["bienc"] += [top_k_retvr]
					
				
				plt.scatter(x_vals["bienc"], y_vals["cur"], c=colors[2][1], label="cur-wo-anc-cost", alpha=0.5,edgecolors=colors[2][0])
				plt.scatter(x_vals["cur"], y_vals["cur"], c=colors[1][1], label="cur", alpha=0.5,edgecolors=colors[1][0])
				plt.scatter(x_vals["bienc"], y_vals["bienc"], c=colors[0][1], label="bienc", alpha=0.5, edgecolors=colors[0][0])
				
				plt.xlim(1, 1100)
				plt.legend()
				plt.grid()
				plt.xlabel("Cost")
				plt.ylabel("Recall")
				curr_plt_file = f"{res_dir}/plots_{mtype}/k={top_k}/recall_vs_cost/{metric}.pdf"
				Path(os.path.dirname(curr_plt_file)).mkdir(exist_ok=True, parents=True)
				plt.savefig(curr_plt_file)
				
				plt.xscale("log")
				curr_plt_file = f"{res_dir}/plots_{mtype}/k={top_k}/recall_vs_cost/{metric}_xlog.pdf"
				plt.savefig(curr_plt_file)
				
				plt.close()
				
			except KeyError as e:
				LOGGER.info(f"Key-error = {e} for mtype = {mtype}, top_k={top_k}, metric={metric}")
				continue
				
	
	except Exception as e:
		embed()
		raise e
		

def main():
	
	data_dir = "../../data/zeshel"
	worlds = get_zeshel_world_info()
	
	parser = argparse.ArgumentParser( description='Run eval for various retrieval methods wrt exact crossencoder scores. This evaluation does not use ground-truth entity information into account')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	
	parser.add_argument("--bi_model_file", type=str, default="", help="File for biencoder ckpt")
	parser.add_argument("--res_dir", type=str, required=True, help="Res dir with score matrices, and to save results")
	parser.add_argument("--n_seeds", type=int, default=10, help="Number of seeds to run")
	parser.add_argument("--plot_only", type=int, default=0, choices=[0,1], help="1 to only plot results, 0 to run exp and then plot results")
	parser.add_argument("--n_ment", type=int, default=100, help="Number of mentions in precomputed mention-entity score matrix")
	parser.add_argument("--batch_size", type=int, default=50, help="Batch size to use with biencoder")
	parser.add_argument("--misc", type=str, default="", help="Misc suffix")
	parser.add_argument("--disable_wandb", type=int, default=0, choices=[0, 1], help="1 to disable wandb and 0 to use it ")
	

	args = parser.parse_args()
	data_name = args.data_name
	
	bi_model_file = args.bi_model_file
	res_dir = args.res_dir
	n_seeds = args.n_seeds
	n_ment = args.n_ment
	batch_size = args.batch_size
	plot_only = bool(args.plot_only)
	disable_wandb = bool(args.disable_wandb)
	misc = "_" + args.misc if args.misc != "" else ""
	
	DATASETS = get_dataset_info(data_dir=data_dir, res_dir=res_dir, worlds=worlds, n_ment=n_ment)
	
	biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file) if bi_model_file != "" else None
	
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(
		base_res_dir=f"{res_dir}/{data_name}/Retrieval_wrt_Exact_CrossEnc",
		data_info=(data_name, DATASETS[data_name]),
		n_seeds=n_seeds,
		batch_size=batch_size,
		plot_only=plot_only,
		biencoder=biencoder,
		disable_wandb=disable_wandb,
		misc=misc,
		arg_dict=args.__dict__
	)


if __name__ == "__main__":
	main()

