import os
import sys
import copy
import json
import pickle
import logging
import argparse
import csv
import glob
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from utils.zeshel_utils import get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def _get_eval_file_from_dir(dir_name, ckpt_type, eval_data, graph_config):
	

	if dir_name.startswith("m=bi_enc"):
		eval_file = f"{dir_name}/eval/method=bienc_{ckpt_type}_d={eval_data}.json"
	elif dir_name.startswith("m=bi_enc"):
		eval_file = f"{dir_name}/eval/method=bienc_{ckpt_type}_d={eval_data}.json"
	elif dir_name.startswith("bienc_6_20"):
		eval_file = f"{dir_name}/method=bienc_d={eval_data}.json"
	elif dir_name.startswith("cur"):
		eval_file = f"{dir_name}/method=cur_d={eval_data}.json"
	elif dir_name.startswith("fixed_anc_ent_cur"):
		eval_file = f"{dir_name}/method=fixed_anc_ent_cur_d={eval_data}.json"
	elif dir_name.startswith("fixed_anc_ent"):
		eval_file = f"{dir_name}/method=fixed_anc_ent_d={eval_data}.json"
	elif dir_name.startswith("tfidf"):
		eval_file = f"{dir_name}/method=tfidf_d={eval_data}.json"
	else:
		raise NotImplementedError(f"Not sure how to handlle dir = {dir_name}")
	
	return eval_file


def _trim_row_name(row_name):
	"""
	Remove additional param in row_val which make no difference to it.
	For eg for CUR, ckpt_vals don't make a difference.
	Similarly, for Bienc, n_ent_anchors don't make a difference
	:param row_name
	:return trimmed_row_name
	"""
	
	try:
		relevant_params = {
			"CUR": ["model", "anc_n_e"],
			"BE": ["model", "ckpt"],
			"TFIDF": ["model"],
			"Fixed_Anc_Ent": ["model", "anc_n_e"],
			"Fixed_Anc_Ent_CUR": ["model", "anc_n_e"],
		}
	
		split_row_name_list = row_name.split("~")
		if "model=CUR" in row_name:
			new_row_name = "~".join([param_n_val if param_n_val.split("=")[0] in relevant_params["CUR"]
									 else f"{param_n_val.split('=')[0]}=None"
									 for param_n_val in split_row_name_list ])
		elif "BE" in row_name:
			new_row_name = "~".join([param_n_val if param_n_val.split("=")[0] in relevant_params["BE"]
									 else f"{param_n_val.split('=')[0]}=None"
									 for param_n_val in split_row_name_list ])
		elif "TFIDF" in row_name:
			new_row_name = "~".join([param_n_val if param_n_val.split("=")[0] in relevant_params["TFIDF"]
									 else f"{param_n_val.split('=')[0]}=None"
									 for param_n_val in split_row_name_list ])
		elif "Fixed_Anc_Ent" in row_name:
			new_row_name = "~".join([param_n_val if param_n_val.split("=")[0] in relevant_params["Fixed_Anc_Ent"]
									 else f"{param_n_val.split('=')[0]}=None"
									 for param_n_val in split_row_name_list ])
		elif "Fixed_Anc_Ent_CUR" in row_name:
			new_row_name = "~".join([param_n_val if param_n_val.split("=")[0] in relevant_params["Fixed_Anc_Ent_CUR"]
									 else f"{param_n_val.split('=')[0]}=None"
									 for param_n_val in split_row_name_list ])
		else:
			new_row_name = row_name
		
		
		return new_row_name
	
	except Exception as e:
		embed()
		raise e


def plot_processed_results(processed_res_fname, res_dir, all_param_vals, fixed_params, var_params, x_axis_params, same_cost_opt):
	
	try:
		
		assert (not same_cost_opt) or (x_axis_params[0] == "top_k_retvr" and x_axis_params[1] == "anc_n_e" and len(x_axis_params) == 2), \
		f" if same_cost_opt = True then x_axis_params contains {x_axis_params} but it should be exactly [top_k_retvr', 'anc_n_e']"
		
		with open(processed_res_fname, "r") as fin:
			"""
			data is three level dict mapping
				fixed-param-vals to
					var-param-vals to
						x-axis-param-vals to
							a scalar metric
			"""
			data = json.load(fin)
			
		LOGGER.info("Processing plots/csvs ")
		all_x_axis_param_vals = [ all_param_vals[param_name] for param_name in x_axis_params]
		for fixed_param_vals in tqdm(data):
			
			curr_data = data[fixed_param_vals]
			# Save a plot or a table for each combination of fixed_param_vals
			
			row_keys = data[fixed_param_vals].keys() # This are various combinations of values taken by var_param_vals eg varying model type
			col_keys = [] # This are keys to x-axis vals eg various k-retrieved values
			for curr_x_axis_param_vals in itertools.product(*all_x_axis_param_vals):
				if same_cost_opt:
					# if "CUR" in row_keys: # For CUR add top_k_retvr and n_ent_anchor vals
					# 	col_keys += [f"cost={curr_x_axis_param_vals[0] + curr_x_axis_param_vals[1]}@"+ "~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])]
					# else: # For other methods just use top_k_retvr vals
					# 	col_keys += [f"cost={curr_x_axis_param_vals[0]}@" + "~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])]
					
					
					# col_keys += [f"cost={curr_x_axis_param_vals[0] + curr_x_axis_param_vals[1]}@" + "~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])]
					# col_keys += [f"cost={curr_x_axis_param_vals[0]}@" + "~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])]
					
					# col_keys += [f"cost={curr_x_axis_param_vals[0] + curr_x_axis_param_vals[1]}"]
					col_keys += [f"cost={curr_x_axis_param_vals[0]}"]
				else:
					col_keys += ["~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])]
				
					
		
			row_name = "~".join(var_params) # Header for row-identifier column
			if same_cost_opt:
				# header = [row_name] + sorted(col_keys, key= lambda x: int(x.split("@")[0][5:])) # In case we add more decriptive breakdown of cost
				col_keys = list(set(col_keys))
				header = [row_name] + sorted(col_keys, key= lambda x: int(x[5:]))
			else:
				header = [row_name] + col_keys
			
			all_row_data = []
			added_row_vals = {}
			for row_val in row_keys: # For each row, get value of all cols i.e. for combinations of x-axis-param-vals
				
				"""
				Remove additional param in row_val which make no difference to it.
				For eg for CUR, ckpt_vals don't make a difference.
				Similarly, for Bienc, n_ent_anchors don't make a difference
				"""
				trimmed_row_val = _trim_row_name(row_val)
				
				if trimmed_row_val in added_row_vals: # Max over all rows that get trimmed to this trimmed_row_val
					row_dict = copy.deepcopy(curr_data[row_val])
					row_dict = {k:f"{v:.2f}" for k,v in row_dict.items() if k in col_keys}
					# Value of unique identifier of this row
					row_dict[row_name] = trimmed_row_val
					
					# Previous Dict for this row
					prev_row_dict = added_row_vals[trimmed_row_val]
					
					max_row_dict = {k:"{:.2f}".format(max(float(v), float(prev_row_dict[k]))) for k,v in row_dict.items() if k in col_keys}
					# Value of unique identifier of this row
					max_row_dict[row_name] = trimmed_row_val
					
					added_row_vals[trimmed_row_val] = max_row_dict
				else:
					# Values of all columns of this row
					row_dict = copy.deepcopy(curr_data[row_val])
					row_dict = {k:f"{v:.2f}" for k,v in row_dict.items() if k in col_keys}
					# Value of unique identifier of this row
					row_dict[row_name] = trimmed_row_val
					
					all_row_data += [row_dict]
					
					added_row_vals[trimmed_row_val] = row_dict
			
			
			all_row_data = added_row_vals.values()
			curr_res_file = f"{res_dir}/{fixed_param_vals}.csv"
			Path(os.path.dirname(curr_res_file)).mkdir(exist_ok=True, parents=True)
			# LOGGER.info(f "Writing results to file = {curr_res_file}")
			with open(curr_res_file, "w") as csvfile:
				
				writer = csv.DictWriter(csvfile, fieldnames=header)
				
				writer.writeheader()
				
				writer.writerows(all_row_data)
				
			

	except Exception as e:
		embed()
		raise e
	





def indexing_time_plot():
	pass


def process_res_for_rq(combined_res, template, all_param_vals, fixed_params, var_params, x_axis_params, val_type, same_cost_opt):
	"""
	Split data from [combined_key] -> [fixed_params][vary_params] format
	:param combined_res:
	:param template:
	:param fixed_params:
	:param var_params:
	:param all_param_vals:
	:return:
	"""
	final_res = defaultdict(lambda :defaultdict(dict))
	
	assert (not same_cost_opt) or (x_axis_params[0] == "top_k_retvr" and x_axis_params[1] == "anc_n_e" and len(x_axis_params) == 2), \
		f" if same_cost_opt = True then x_axis_params contains {x_axis_params} but it should be exactly [top_k_retvr', 'anc_n_e']"
	
	all_fixed_param_vals = [ all_param_vals[param_name] for param_name in fixed_params]
	all_var_param_vals = [ all_param_vals[param_name] for param_name in var_params]
	all_x_axis_param_vals = [ all_param_vals[param_name] for param_name in x_axis_params]
	
	for curr_fixed_param_vals in itertools.product(*all_fixed_param_vals):
		fixed_key = "~".join([ f"{param}={param_val}" for param, param_val in zip(fixed_params, curr_fixed_param_vals)])
		# eg fixed_key = f"graph_metric={graph_metric}~entry_method={entry_method}~crossenc={crossenc}~bienc={bienc}~n_ment={n_ment}"
		for curr_var_param_vals in itertools.product(*all_var_param_vals):
			# eg var_key = f"graph_type={graph_type}~embed_type={embed_type}"
			var_key = "~".join([ f"{param}={param_val}" for param, param_val in zip(var_params, curr_var_param_vals)])
			
			for curr_x_axis_param_vals in itertools.product(*all_x_axis_param_vals):
				if same_cost_opt:
					if "CUR" in var_key: # For CUR add top_k_retvr and n_ent_anchor vals
						x_axis_key = f"cost={curr_x_axis_param_vals[0] + curr_x_axis_param_vals[1]}"
					elif "Fixed_Anc_Ent" in var_key: # For Fixed_Anc_Ent, total cost is top_k_retvr and nm_train (i.e. number of row of ent2ent matrix used for indexing)
						if "nm_train" in fixed_params:
							nm_train_val = curr_fixed_param_vals[fixed_params.index("nm_train")]
						elif "nm_train" in var_params:
							nm_train_val = curr_var_param_vals[var_params.index("nm_train")]
						else:
							raise NotImplementedError(f"nm_train should be in fixed or var params if same_cost_opt is True")
						x_axis_key = f"cost={curr_x_axis_param_vals[0]}"
					else: # For other methods just use top_k_retvr vals
						x_axis_key = f"cost={curr_x_axis_param_vals[0]}"
				else:
					x_axis_key = "~".join([ f"{param}={param_val}" for param, param_val in zip(x_axis_params, curr_x_axis_param_vals)])
				
				comb_param_dict = {param_name:param_val for param_name, param_val in zip(fixed_params, curr_fixed_param_vals)}
				comb_param_dict.update({param_name:param_val for param_name, param_val in zip(var_params, curr_var_param_vals)})
				comb_param_dict.update({param_name:param_val for param_name, param_val in zip(x_axis_params, curr_x_axis_param_vals)})
				comb_key = template.format(
					**comb_param_dict
				)
				
				if comb_key not in combined_res: continue
				
				if x_axis_key in final_res[fixed_key][var_key]: # Take best possible value if there are multiple values for this x_axis_key
					final_res[fixed_key][var_key][x_axis_key] = max(final_res[fixed_key][var_key][x_axis_key], combined_res[comb_key][val_type])
				else:
					final_res[fixed_key][var_key][x_axis_key] = combined_res[comb_key][val_type]

	
	return final_res


def create_combine_result_file(
		base_res_dir, data_name, model_vals, n_ment_anchors_vals, split_idx_vals, eval_data_vals,
		ckpt_type_vals, n_ent_anchors_vals, top_k_vals, top_k_retr_vals_base, top_k_retr_vals_cur, graph_config_vals
):
	try:
		
		"""
		Define a key for combining various params then follow graph eval like process to plot or tabulate result while varying one param and keeping other frozen.
		
		Parameters
		1. Model
		2. Number of anchor mentions
		3. Top-k Retrieved
		4. Top-k for eval
		5. Split_Idx
		
		Eval specific
		
		5. Number of anchor entities for CUR
		6. ckpt type
		7. eval data
		
	
		"""
		
		combined_res = defaultdict(dict)
		file_read = 0
		for model_key, n_ment_anchors, split_idx, eval_data, ckpt_type, graph_config in \
				itertools.product(model_vals, n_ment_anchors_vals, split_idx_vals, eval_data_vals, ckpt_type_vals, graph_config_vals):
			
			file_name = _get_eval_file_from_dir(
				dir_name=model_vals[model_key],
				ckpt_type=ckpt_type,
				eval_data=eval_data,
				graph_config=graph_config
			)
			file_name = f"{base_res_dir}/{data_name}/nm_train={n_ment_anchors}/split_idx={split_idx}/{file_name}"
			
			if not os.path.isfile(file_name):
				LOGGER.info(f"file not present : {file_name}")
				continue
			
			with open(file_name, "r") as fin:
				eval_res = json.load(fin)
				file_read += 1
			
			
			top_k_retr_vals = top_k_retr_vals_cur if "CUR" in model_key else top_k_retr_vals_base
			for n_ent_anchors, top_k_retvr, top_k in itertools.product(n_ent_anchors_vals, top_k_retr_vals, top_k_vals):
				if top_k_retvr < top_k: continue
				if top_k_retvr > NUM_ENTS[data_name]: continue
			
				try:
					key = f"nm_train={n_ment_anchors}~split_idx={split_idx}~top_k_retvr={top_k_retvr}~top_k={top_k}~model={model_key}~ckpt={ckpt_type}~anc_n_e={n_ent_anchors}~data_type={eval_data}~graph_config={graph_config}"
					val = 100*eval_res[f"seed={0}"][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}_anc_n_e={n_ent_anchors}"]["exact_vs_reranked_approx_retvr~common_frac_mean"]
					
					# Use correct checkpoint for train and test data
					if ckpt_type == "best_wrt_dev" and eval_data == "train": continue
					if ckpt_type == "eoe" and eval_data == "test": continue
					
					combined_res[key]["prec@k"] = val
				except Exception as e:
					# LOGGER.info(f"Exception = {e} raised for -> nm_train={n_ment_anchors}~split_idx={split_idx}~top_k_retvr={top_k_retvr}~top_k={top_k}~m={model_key}~ckpt={ckpt_type}~anc_n_e={n_ent_anchors}~data={eval_data}~graph_config={graph_config}")
					continue
		
		
		LOGGER.info(f"Writing combined result file after combining {file_read} files")
		combined_res_file = f"{base_res_dir}/{data_name}/combined_res_file.json"
		with open(combined_res_file, "w") as fout:
			json.dump(combined_res, fout, indent=4)
		
		return combined_res_file
	except Exception as e:
		embed()
		raise e
	


def run(data_name, base_res_dir):
	
	model_vals = {
		"TFIDF":"tfidf_s=1",
		"01_D_BE_TRP_100" 	: "m=bi_enc_l=ce_neg=top_ce_w_bienc_hard_negs_trp_s=1234_topk_100_from_6_20_bienc",
		"03_D_BE_MATCH_100"	: "m=bi_enc_l=ce_neg=top_ce_match_s=1234_topk_100_from_6_20_bienc",
		
		"01_S_BE_TRP_100"	: "m=bi_enc_l=ce_neg=top_ce_w_bienc_hard_negs_trp_s=1234_topk_100_from_scratch",
		"03_S_BE_MATCH_100"	: "m=bi_enc_l=ce_neg=top_ce_match_s=1234_topk_100_from_scratch",
		
		"00_6_20_BE": "bienc_6_20",
		
		"CUR":"cur_s=1",
		"Fixed_Anc_Ent": "fixed_anc_ent_s=1",
		"Fixed_Anc_Ent_CUR": "fixed_anc_ent_cur_s=1",
	
	}
	
	

	n_ment_anchors_vals = [100, 500, 2000] if data_name != "pro_wrestling" else [100, 500, 1000]
	split_idx_vals = [0]
	eval_data_vals = ["train", "test"]
	# eval_data_vals = ["test"]
	ckpt_type_vals = ["best_wrt_dev", "eoe"]
	# ckpt_type_vals = ["best_wrt_dev"]

	n_ent_anchors_vals_base = [10, 50, 100, 200, 500, 1000, 2000]
	n_ent_anchors_vals = [v for v in n_ent_anchors_vals_base if v < NUM_ENTS[data_name]] + [NUM_ENTS[data_name]]

	top_k_vals = [1, 10, 50, 100]
	top_k_retr_vals_base = [1, 10, 50, 100, 200, 500, 1000]
	top_k_retr_vals_cur = top_k_retr_vals_base + [x - y for x in top_k_retr_vals_base for y in n_ent_anchors_vals if x - y > 0]
	top_k_retr_vals_cur +=  [int(k*frac) for k in top_k_retr_vals_base for frac in np.arange(0.1, 1.0, 0.1)]
	n_ent_anchors_vals =   sorted(list(set(n_ent_anchors_vals + top_k_retr_vals_cur)))
	
	graph_config_vals = [""]
	
	
	comb_res_file = create_combine_result_file(
		base_res_dir=base_res_dir,
		data_name=data_name,
		model_vals=model_vals,
		n_ment_anchors_vals=n_ment_anchors_vals,
		split_idx_vals=split_idx_vals,
		eval_data_vals=eval_data_vals,
		ckpt_type_vals=ckpt_type_vals,
		n_ent_anchors_vals=n_ent_anchors_vals,
		top_k_vals=top_k_vals,
		top_k_retr_vals_base=top_k_retr_vals_base,
		top_k_retr_vals_cur=top_k_retr_vals_cur,
		graph_config_vals=graph_config_vals
	)
	
	
	
	################################# PROCESS AND CREATE TABLES/PLOTS FOR VARIOUS RQS ##################################
	template =  "nm_train={nm_train}~split_idx={split_idx}~top_k_retvr={top_k_retvr}~top_k={top_k}~model={model}~ckpt={ckpt}~anc_n_e={anc_n_e}~data_type={data_type}~graph_config={graph_config}"
	all_param_vals = {
		"nm_train": n_ment_anchors_vals,
		"split_idx": split_idx_vals,
		"top_k_retvr": top_k_retr_vals_cur,
		"top_k": top_k_vals,
		"model": model_vals,
		"ckpt": ckpt_type_vals,
		"anc_n_e": n_ent_anchors_vals,
		"data_type": eval_data_vals,
		"graph_config": graph_config_vals
	}
	
	all_param_vals_same_cost = copy.deepcopy(all_param_vals)
	all_param_vals_same_cost["top_k_retvr"] = top_k_retr_vals_base
	
	LOGGER.info(f"Reading combined res file from {comb_res_file}")
	with open(comb_res_file, "r") as fin:
		combined_res = json.load(fin)


	RQs = {
		"RQ1_Model_Performance_At_Equal_Num_Retrieved":{
			"var_params": ["model", "ckpt", "anc_n_e", "graph_config"],
			"fixed_params": ["nm_train", "split_idx", "top_k", "data_type"],
			"x_axis_params": ["top_k_retvr"],
			"val_type": "prec@k",
			"same_cost_opt": False
		},
		"RQ2_Model_Performance_At_Equal_Test_Cost":{
			"var_params": ["model", "ckpt", "graph_config"],
			"fixed_params": ["nm_train", "split_idx", "top_k", "data_type"],
			"x_axis_params": ["top_k_retvr", "anc_n_e"],
			"val_type": "prec@k",
			"same_cost_opt": True
		},
		# "RQ3_Model_Performance_As_Train_Data_Increases":{
		# 	"var_params": ["model", "ckpt", "anc_n_e", "graph_config"],
		# 	"fixed_params": ["split_idx", "top_k", "data_type", "top_k_retvr"],
		# 	"x_axis_params": ["nm_train"],
		# 	"val_type": "prec@k",
		# 	"same_cost_opt": False
		# },
		# "RQ2_Model_Performance_At_Equal_Test_Cost_Separate":{
		# 	"var_params": ["model", "ckpt"],
		# 	"fixed_params": ["nm_train", "split_idx", "top_k", "data_type"],
		# 	"x_axis_params": ["top_k_retvr", "anc_n_e"],
		# 	"val_type": "prec@k",
		# 	"same_cost_opt": False
		# }
		
	}
	
	
	for curr_rq in RQs:
		LOGGER.info(f"Processing data for RQ : {curr_rq}")

		processed_res = process_res_for_rq(
			template=template,
			combined_res=combined_res,
			all_param_vals=all_param_vals,
			fixed_params=RQs[curr_rq]["fixed_params"],
			var_params=RQs[curr_rq]["var_params"],
			x_axis_params=RQs[curr_rq]["x_axis_params"],
			val_type=RQs[curr_rq]["val_type"],
			same_cost_opt=RQs[curr_rq]["same_cost_opt"]
		)
		

		# Save data
		curr_rq_res_dir = f"{base_res_dir}/{data_name}/RQs/{curr_rq}"
		process_fname = f"{curr_rq_res_dir}/processed_res.json"
		Path(os.path.dirname(process_fname)).mkdir(exist_ok=True, parents=True)
		with open(process_fname, "w") as fout:
			json.dump(processed_res, fout, indent=4)
		
		# Plot data or convert into table
		plot_processed_results(
			processed_res_fname=process_fname,
			res_dir=f"{curr_rq_res_dir}/plots",
			# all_param_vals=all_param_vals if (not RQs[curr_rq]["same_cost_opt"]) else all_param_vals_same_cost, # Need to use all_param_vals_same_cost here as we only want to look at a table with just top_k_retvr_values = top_k_retr_vals_base
			all_param_vals=all_param_vals_same_cost, # Need to use all_param_vals_same_cost here as we only want to look at a table with just top_k_retvr_values = top_k_retr_vals_base
			fixed_params=RQs[curr_rq]["fixed_params"],
			var_params=RQs[curr_rq]["var_params"],
			x_axis_params=RQs[curr_rq]["x_axis_params"],
			same_cost_opt=RQs[curr_rq]["same_cost_opt"]
		)
	
	
	####################################################################################################################
	

def main():
	
	base_res_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/models"
	
	worlds = get_zeshel_world_info()

	parser = argparse.ArgumentParser(description='Compile results for various retrieval methods wrt exact crossencoder scores')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds], help="Dataset name")
	
	args = parser.parse_args()

	data_name = args.data_name
	
	LOGGER.info(f"Running inference for world = {data_name}")
	run(data_name=data_name, base_res_dir=base_res_dir)


if __name__ == "__main__":
	main()

