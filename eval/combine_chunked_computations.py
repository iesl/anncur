import os
import sys
import json
import torch
import pickle
import logging
import numpy as np
from IPython import embed
from pathlib import Path
from utils.zeshel_utils import N_ENTS_ZESHEL, N_MENTS_ZESHEL
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def combine_bi_plus_cross_eval_results():
	try:
		topk = 64
		# dataset_name = "starwars"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,8001,2000)
		# ]
		# dir_list += [f"m=1824_k={topk}_1_eoe-0-last.ckpt_mstart_10000"]
		
		# dataset_name = "military"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,10001,2000)
		# ]
		# dir_list += [f"m=1063_k={topk}_1_eoe-0-last.ckpt_mstart_12000"]
		
		# dataset_name = "doctor_who"
		# dir_list = [
		# 	f"m=3000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,3001,3000)
		# ]
		# dir_list += [f"m=2334_k={topk}_1_eoe-0-last.ckpt_mstart_6000"]
		#
		# dataset_name = "american_football"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1,10)
		# ]
		# dir_list += [f"m=1898_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		# dataset_name = "final_fantasy"
		# dir_list = [
		# 	f"m=3000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		# ]
		# dir_list += [f"m=3041_k={topk}_1_eoe-0-last.ckpt_mstart_3000"]
		
		# dataset_name = "elder_scrolls"
		# dir_list = [
		# 	f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		# ]
		# dir_list += [f"m=2275_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		dataset_name = "fallout"
		dir_list = [
			f"m=2000_k={topk}_1_eoe-0-last.ckpt_mstart_{mstart}" for mstart in range(0,1)
		]
		dir_list += [f"m=1286_k={topk}_1_eoe-0-last.ckpt_mstart_2000"]
		
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_for_hard_neg_training/eval/{dataset_name}"
		file_list = [f"{res_dir}/{curr_dir}/crossenc_topk_preds_w_bienc_retrvr.txt" for curr_dir in dir_list]

		
		combined_topk_preds = {"indices":[], "scores":[]}
		for curr_file in file_list:
			with open(curr_file, "r") as fin:
				curr_topk_preds = json.load(fin)
				
				combined_topk_preds["indices"] += curr_topk_preds["indices"]
				combined_topk_preds["scores"] += curr_topk_preds["scores"]
				LOGGER.info(f"curr_file {curr_file}")
				LOGGER.info(f"Number of rows in indices : {len(curr_topk_preds['indices'])}")
				LOGGER.info(f"Number of rows in scores  : {len(curr_topk_preds['scores'])}")
		
		LOGGER.info(f"Final number of rows in indices : {len(combined_topk_preds['indices'])}")
		LOGGER.info(f"Final number of rows in scores  : {len(combined_topk_preds['scores'])}")
		
		
		# temp_file = f"{res_dir}/m=-1_k=500_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		# with open(temp_file, "rb") as fin:
		# 	existing_res = json.load(fin)
		#
		#
		# for ment_idx, (exist_row, comb_row) in enumerate(zip(existing_res["indices"], combined_topk_preds["indices"])):
		# 	exist_row_set = set(exist_row[:150])
		# 	comb_row_set = set(comb_row[:150])
		# 	if len(set(exist_row_set) - set(comb_row_set)) != 0 or len(set(comb_row_set) - set(exist_row_set)) != 0:
		# 		LOGGER.info(f"{ment_idx} E-C: {set(exist_row_set) - set(comb_row_set)}")
		# 		LOGGER.info(f"{ment_idx} C-E: {set(comb_row_set) - set(exist_row_set)}\n")
		# 		for x in exist_row_set-comb_row_set:
		# 			_idx = existing_res["indices"][ment_idx].index(x)
		# 			score = existing_res["scores"][ment_idx][_idx]
		# 			LOGGER.info(f"E-C: Scores : {_idx}, {x} ->{score}")
		#
		# 		for x in comb_row_set-exist_row_set:
		# 			_idx = combined_topk_preds["indices"][ment_idx].index(x)
		# 			score = combined_topk_preds["scores"][ment_idx][_idx]
		# 			LOGGER.info(f"C-E: Scores : {_idx}, {x}->{score}")
		#
		# embed()
		
		comb_file = f"{res_dir}/m=-1_k={topk}_1_eoe-0-last.ckpt/crossenc_topk_preds_w_bienc_retrvr.txt"
		Path(os.path.dirname(comb_file)).mkdir(exist_ok=True, parents=True)
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1 "))
			if not over_write: return
			
		LOGGER.info(f"Writing result to file : {comb_file}")
		
		with open(comb_file, "w") as fout:
			json.dump(combined_topk_preds, fout)
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e


def combine_m2e_eval_results():
	try:
		# "joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/score_mats_model-3-24599.0--75.46.ckpt"
		#
		# "joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_from_scratch/score_mats_model-3-19679.0--77.23.ckpt"
		#
		# "m=cross_enc_l=ce_neg=bienc_distill_s=1234_crossenc_w_embeds/model/score_mats_model-1-12279.0-1.91.ckpt"
		#
		
		# res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/joint_train/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/score_mats_model-3-24599.0--75.46.ckpt"
		res_dir = f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt"
		# res_dir = f"../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_cls_w_lin_tfidf_hard_negs/score_mats_model-1-12279.0--90.95.ckpt"
		
		
		# dataset_name = "doctor_who"
		#
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{2100}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"]
		# step = 100
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(2100, 3901, step)]
		#
		dataset_name = "yugioh"
		step = 100
		file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
					 for mstart in range(0, 3201, step)]
		file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{74}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{3350}.pkl"]
		#
		# dataset_name = "pro_wrestling"
		# step = 250
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(0, 1001, step)]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{142}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{1250}.pkl"]
		
		# dataset_name = "military"
		#
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{1000}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"]
		#
		# step = 50
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1000, 1201, step)]
		#
		# step = 1
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1250, 1300, step)]
		#
		# step = 50
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1300, 1901, step)]
		# step = 10
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(1950, 2491, step)]
		#
		# dataset_name = "star_trek"
		# step = 50
		# file_list = [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{2500}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{step}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{mstart}.pkl"
		# 			 for mstart in range(2500, 4151, step)]
		# file_list += [f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{27}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_Falsemstart_{4200}.pkl"]
		
		comb_ment_to_ent_scores_list = []
		comb_test_data_list = []
		comb_mention_tokens_list = []
		
		# Values that should remain fixed across all files
		comb_entity_id_list = np.arange(N_ENTS_ZESHEL[dataset_name])
		comb_entity_tokens_list = None
		comb_arg_dict = []
		
		for curr_file in file_list:
			with open(curr_file, "rb") as fin:
				dump_dict = pickle.load(fin)
				ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
				test_data = dump_dict["test_data"]
				mention_tokens_list = dump_dict["mention_tokens_list"]
				entity_id_list = dump_dict["entity_id_list"]
				entity_tokens_list = dump_dict["entity_tokens_list"]
				arg_dict = dump_dict["arg_dict"]
			
			comb_ment_to_ent_scores_list += [ment_to_ent_scores]
			comb_test_data_list += [test_data]
			comb_mention_tokens_list += [mention_tokens_list]
			
			comb_arg_dict += [arg_dict]
			
			assert comb_entity_tokens_list is None or (comb_entity_tokens_list == entity_tokens_list).all()
			assert (comb_entity_id_list == entity_id_list).all()
			
			LOGGER.info(f"Shape of current m2e matrix : {ment_to_ent_scores.shape}")
			
		comb_ment_to_ent_scores = torch.cat(comb_ment_to_ent_scores_list)
		comb_test_data = [x for xs in comb_test_data_list for x in xs] # Concat lists present in the list
		comb_mention_tokens = [x for xs in comb_mention_tokens_list for x in xs]
		
		LOGGER.info(f"Shape of final m2e matrix : {comb_ment_to_ent_scores.shape}")
		total_n_ments = comb_ment_to_ent_scores.shape[0]
		assert total_n_ments == len(comb_test_data), f"total_n_ments = {total_n_ments} != len(comb_test_data) = {len(comb_test_data)}"
		assert total_n_ments == len(comb_mention_tokens), f"total_n_ments = {total_n_ments} != len(comb_test_data) = {len(comb_test_data)}"
		
		comb_file = f"{res_dir}/{dataset_name}/ment_to_ent_scores_n_m_{total_n_ments}_n_e_{N_ENTS_ZESHEL[dataset_name]}_all_layers_False.pkl"
		
		
		if os.path.exists(comb_file):
			LOGGER.info(f"File exists = {comb_file}")
			over_write = int(input("Overwrite? 0 or 1\n"))
			if not over_write: return

		LOGGER.info(f"Writing result to file : {comb_file}")
		with open(comb_file, "wb") as fout:
			
			res = {
				"ment_to_ent_scores":comb_ment_to_ent_scores,
				"ment_to_ent_scores.shape":comb_ment_to_ent_scores.shape,
				"test_data":comb_test_data,
				"mention_tokens_list":comb_mention_tokens,
				"entity_id_list":comb_entity_id_list,
				"entity_tokens_list":comb_entity_tokens_list,
				"arg_dict":comb_arg_dict[-1]
			}
			pickle.dump(res, fout)
			
		
		
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()
		raise e



def main():
	# combine_bi_plus_cross_eval_results()
	# combine_m2e_eval_results()
	pass


if __name__ == "__main__":
	main()
