import os
import sys
import json
import glob
import time
import logging
import itertools
from pathlib import Path

from utils.config import Config
from utils.zeshel_utils import N_ENTS_ZESHEL, get_zeshel_world_info

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s -%(funcName)20s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _get_param_config():
	
	base_precomp_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/6_256_e_crossenc/score_mats_model-2-15999.0--79.46.ckpt"
	
	# data_name_vals = ["yugioh", "star_trek", "military", "doctor_who"]
	# data_name_vals = ["star_trek", "military", "doctor_who"]
	data_name_vals = ["yugioh"]
	# data_name_vals = ["star_trek", "military"]
	# data_name_vals = ["pro_wrestling"]
	# data_name_vals =
	
	neg_strategy_vals = ["top_ce_w_bienc_hard_negs_trp", "top_ce_match_25", "top_ce_match_100"]
	# neg_strategy_vals = ["top_ce_match_25", "top_ce_match_100"]
	# neg_strategy_vals = ["top_ce_w_bienc_hard_negs_trp", "top_ce_match_25"]
	# neg_strategy_vals = ["top_ce_match_100"]
	# neg_strategy_vals = ["top_ce_match"]
	
	loss_func_vals = ["ce"]
	bienc_param_vals = {
		"6_20_bienc" : "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt",
		"scratch" : "None",
	}
	
	# num_train_vals = [100, 500, 1000] # only for pro_wrestling
	# assert len(data_name_vals) == 1 and "pro_wrestling" in data_name_vals
	
	num_train_vals = [100, 500, 2000]
	assert "pro_wrestling" not in data_name_vals
	
	split_num_vals = [0]
	num_epoch_vals = [10]
	
	neg_strategy_specific_params = {
		"top_ce_match_25":{
			"distill_n_labels": 25,
			"train_batch_size": 4,
			"eval_batch_size": 16,
			"reload_dataloaders_every_n_epochs": 0,
		},
		"top_ce_match_100":{
			"distill_n_labels": 100,
			"train_batch_size": 4,
			"eval_batch_size": 16,
			"reload_dataloaders_every_n_epochs": 0,
		},
		"top_ce_w_bienc_hard_negs_trp":{
			"distill_n_labels": 100,
			"train_batch_size": 32,
			"eval_batch_size": 256,
			"reload_dataloaders_every_n_epochs": 1,
		},
	}

	
	all_run_configs = []
	all_param_configs = []
	missing_data_configs = []
	for data_name, split_num, loss_func, neg_strategy, bienc_key, num_epoch, num_train in \
			itertools.product(data_name_vals, split_num_vals, loss_func_vals, neg_strategy_vals, bienc_param_vals, num_epoch_vals, num_train_vals):
		
		neg_strategy_trimmed = neg_strategy[:12] if neg_strategy.startswith("top_ce_match") else neg_strategy
		misc = f"topk_{neg_strategy_specific_params[neg_strategy]['distill_n_labels']}_from_{bienc_key}"
		
		curr_run_params = {
			"exp_id": "8_CUR_EMNLP",
			"res_dir_prefix" :  f"models/{data_name}/nm_train={num_train}/split_idx={split_num}/",
			"train_domains": data_name,
			"dev_domains": data_name,
			"loss_type": loss_func,
			"neg_strategy": neg_strategy_trimmed,
			
			"distill_n_labels": neg_strategy_specific_params[neg_strategy]["distill_n_labels"],
			"train_batch_size": neg_strategy_specific_params[neg_strategy]["train_batch_size"],
			"grad_acc_steps": min(neg_strategy_specific_params[neg_strategy]["train_batch_size"], 4), # Set to min of train_batch_size and 4
			"reload_dataloaders_every_n_epochs": neg_strategy_specific_params[neg_strategy]["reload_dataloaders_every_n_epochs"],
			"eval_batch_size": neg_strategy_specific_params[neg_strategy]["eval_batch_size"],
			
			"train_ent_w_score_file_template" : f"{base_precomp_dir}/{data_name}/m2e_splits/nm_train={num_train}/split_idx={split_num}/train_train.pkl",
			"dev_ent_w_score_file_template" : f"{base_precomp_dir}/{data_name}/m2e_splits/nm_train={num_train}/split_idx={split_num}/train_dev.pkl",
			"path_to_model": bienc_param_vals[bienc_key],
			"num_epochs": num_epoch,
			
			"warmup_proportion": 0.00,
			"eval_interval": 0.2,
			"strategy": "ddp",
			"num_gpus": 2,
			"num_top_k_ckpts": 1,
			
			"misc": misc
		}
		
		curr_param_config = {
			"data_name":data_name,
			"num_train":num_train,
			"split_num":split_num,
			"loss_func":loss_func,
			"neg_strategy":neg_strategy_trimmed,
			"bienc_key":bienc_key,
			"num_epoch":num_epoch,
			"misc":misc,
			"res_dir_prefix": f"models/{data_name}/nm_train={num_train}/split_idx={split_num}/",
		}
		# Train batch size should be a multiple of grad_acc_steps. And grad_acc_steps = min(4, train_batch_size)
		assert neg_strategy_specific_params[neg_strategy]["train_batch_size"] % 4 == 0 or neg_strategy_specific_params[neg_strategy]["train_batch_size"] < 4
		assert " " not in data_name and "\t" not in data_name, f"data_name = {data_name} should not have any white spaces. If passing multiple datanames then we should also fix train_ent_w_score_file_template files"
		
		# if not os.path.isfile(curr_run_params["train_ent_w_score_file_template"]):
		# 	LOGGER.info(f"train_ent_w_score_file_template file does not exist : {curr_run_params['train_ent_w_score_file_template']} ")
		# 	missing_data_configs += [curr_run_params]
		# 	continue
		#
		# if not os.path.isfile(curr_run_params["dev_ent_w_score_file_template"]):
		# 	LOGGER.info(f"dev_ent_w_score_file_template file does not exist : {curr_run_params['dev_ent_w_score_file_template']} ")
		# 	missing_data_configs += [curr_run_params]
		# 	continue

		all_run_configs += [curr_run_params]
		all_param_configs += [curr_param_config]
	
	LOGGER.info(f"{len(missing_data_configs)} configs have missing data files")
	LOGGER.info(f"Returning {len(all_run_configs)} configs")
	
	return all_run_configs, all_param_configs


def launch_train_jobs():
	
	launch_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else False
	allow_running_even_if_already_run = int(sys.argv[2]) if len(sys.argv) > 2 else False
	print_command = int(sys.argv[3]) if len(sys.argv) > 3 else False
	
	all_configs, _ = _get_param_config()
	
	all_commands = []
	previously_run_commands = []
	found = 0
	default_config_file = "config/ce_distill/zeshel_bi_enc_distill.json"
	default_config = Config(default_config_file)
	
	for ctr, curr_config in enumerate(all_configs):
		# curr_command = f"sbatch -p 2080ti-long --gres gpu:2 --mem 64GB --job-name bd_{curr_config['train_domains']}_{ctr} bin/run.sh python models/train.py "
		curr_command = f"sbatch -p rtx8000-long --gres gpu:2 --mem 64GB --job-name bd_{curr_config['train_domains']}_{ctr} bin/run.sh python models/train.py " # FIXME: Change rtx8000
		curr_command += f"--config  {default_config_file} "
		for key,val in curr_config.items():
			curr_command += f" --{key} {val} "
		
		result_dir = "{base}/d={d}/{prefix}m={m}_l={l}_neg={neg}_s={s}{misc}".format(
			base="../../results/" + curr_config["exp_id"],
			prefix=curr_config["res_dir_prefix"],
			d=curr_config.get("data_type", default_config.data_type),
			m=curr_config.get("model_type", default_config.model_type),
			l=curr_config["loss_type"],
			neg=curr_config["neg_strategy"],
			s=curr_config.get("seed", default_config.seed),
			misc="_{}".format(curr_config["misc"]) if curr_config["misc"] != "" else "")
		
		# LOGGER.info(f"curr_res_dir: {result_dir}")
		if allow_running_even_if_already_run or not os.path.isdir(result_dir):
			all_commands += [curr_command]
			if launch_jobs: os.system(command=curr_command)
			if print_command: print(f"{curr_command}")
			# LOGGER.info(f"curr_res_dir: {result_dir}")
		else:
			previously_run_commands += [curr_command]
	
	LOGGER.info(f"Previously run commands = {len(previously_run_commands)}")
	LOGGER.info(f"Commands run now = {len(all_commands)}")
	LOGGER.info(f"Found = {found}")
	

def get_bienc_eval_job_configs(base_data_dir, ckpt_type, eval_on_train_data):
	
	
	
	batch_size = 50
	default_config_file = "config/ce_distill/zeshel_bi_enc_distill.json"
	default_config = Config(default_config_file)
	
	all_train_configs, all_train_param_configs = _get_param_config()
	
	
	all_eval_configs = []
	for ctr, curr_config in enumerate(all_train_configs):

		result_dir = "{base}/d={d}/{prefix}m={m}_l={l}_neg={neg}_s={s}{misc}".format(
			base="../../results/" + curr_config["exp_id"],
			prefix=curr_config["res_dir_prefix"],
			d=curr_config.get("data_type", default_config.data_type),
			m=curr_config.get("model_type", default_config.model_type),
			l=curr_config["loss_type"],
			neg=curr_config["neg_strategy"],
			s=curr_config.get("seed", default_config.seed),
			misc="_{}".format(curr_config["misc"]) if curr_config["misc"] != "" else "")
		
		data_name = all_train_param_configs[ctr]['data_name']
		nm_train = all_train_param_configs[ctr]['num_train']
		split_idx = all_train_param_configs[ctr]['split_num']
		misc = all_train_param_configs[ctr]['misc']
		
		
		if ckpt_type == "eoe":
			num_epoch = all_train_param_configs[ctr]["num_epoch"]
			
			epoch_ctr = 1
			bi_model_file = f"{result_dir}/model/eoe-{num_epoch-epoch_ctr}-last.ckpt" # TODO: Also try to best wrt dev
			while not os.path.isfile(bi_model_file) and num_epoch-epoch_ctr >= 0:
				bi_model_file = f"{result_dir}/model/eoe-{num_epoch-epoch_ctr}-last.ckpt"
				epoch_ctr +=1
			
			if not os.path.isfile(bi_model_file):
				LOGGER.info(f"bi_model_file not found = {bi_model_file}")
		elif ckpt_type == "best_wrt_dev":
			file_list = glob.glob(f"{result_dir}/model/model-*.ckpt")
			file_list = sorted( file_list, key=os.path.getmtime, reverse=True)
			if len(file_list) > 0:
				bi_model_file = file_list[0] # Get most recent checkpoint
			else:
				LOGGER.info(f"No checkpoint in dir = {result_dir}")
				continue
		else:
			raise NotImplementedError(f"Checkpoint type = {ckpt_type} not supported")
			
		
		train_data_file = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train.pkl"
		if eval_on_train_data:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train_train.pkl"
		else:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/test.pkl"
		
		eval_config = {
			"data_name": data_name,
			"eval_method": "bienc",
			"res_dir": f"{result_dir}/eval",
			"test_data_file": test_data_file,
			"train_data_file": train_data_file,
			
			"bi_model_file": bi_model_file,
			"batch_size": batch_size,
			"misc": ckpt_type + "_d=train" if eval_on_train_data else ckpt_type + "_d=test",
			
			
			# "n_seeds": n_seeds, Not needed for bienc eval
		}
		
		all_eval_configs += [eval_config]
		
	return all_eval_configs, all_train_configs, all_train_param_configs
	

def get_cur_eval_job_configs(base_data_dir, base_res_dir, eval_on_train_data):
	
	n_seeds = 1
	all_train_configs, all_train_param_configs = _get_param_config()
	
	all_eval_configs = []
	processed_res_dirs = {}
	for ctr, curr_config in enumerate(all_train_configs):

		data_name = all_train_param_configs[ctr]['data_name']
		nm_train = all_train_param_configs[ctr]['num_train']
		split_idx = all_train_param_configs[ctr]['split_num']
		# misc = all_train_param_configs[ctr]['misc']
		
		train_data_file = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train.pkl"
		if eval_on_train_data:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train_train.pkl"
		else:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/test.pkl"
		
		
		result_dir      = f"{base_res_dir}/models/{data_name}/nm_train={nm_train}/split_idx={split_idx}/cur_s={n_seeds}"
		
		# There are multiple train configs that train on the exact same data so avoid running cur eval multiple times on the same test data
		if result_dir in processed_res_dirs: continue
		
		processed_res_dirs[result_dir] = 1
		eval_config = {
			"data_name": data_name,
			"eval_method": "cur",
			"res_dir": result_dir,
			"test_data_file": test_data_file,
			"train_data_file": train_data_file,
			
			"n_seeds": n_seeds,
		
			"misc": ("d=train" if eval_on_train_data else "d=test")
			
			# "bi_model_file": bi_model_file, :Not needed for cur eval
			# "batch_size": batch_size, :Not needed for cur eval
		}
		
		all_eval_configs += [eval_config]
		
	return all_eval_configs, all_train_configs, all_train_param_configs


def get_fixed_anc_ent_eval_job_configs(eval_method, base_data_dir, base_res_dir, eval_on_train_data):
	
	assert eval_method in ["fixed_anc_ent", "fixed_anc_ent_cur"], f"eval method = {eval_method} not supported"
	n_seeds = 1
	all_train_configs, all_train_param_configs = _get_param_config()
	
	all_eval_configs = []
	processed_res_dirs = {}
	for ctr, curr_config in enumerate(all_train_configs):

		data_name = all_train_param_configs[ctr]['data_name']
		nm_train = all_train_param_configs[ctr]['num_train']
		split_idx = all_train_param_configs[ctr]['split_num']
		# misc = all_train_param_configs[ctr]['misc']
		
		train_data_file = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train.pkl"
		if eval_on_train_data:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train_train.pkl"
		else:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/test.pkl"
		
		
		max_n_anc_ent = 1000
		e2e_fname =  f"{base_data_dir}/{data_name}/ent_to_ent_scores_n_e_{N_ENTS_ZESHEL[data_name]}x{N_ENTS_ZESHEL[data_name]}_topk_{max_n_anc_ent}_embed_none_m2e_.pkl"
		result_dir = f"{base_res_dir}/models/{data_name}/nm_train={nm_train}/split_idx={split_idx}/{eval_method}_s={n_seeds}"
		
		# There are multiple train configs that train on the exact same data so avoid running cur eval multiple times on the same test data
		if result_dir in processed_res_dirs: continue
		
		processed_res_dirs[result_dir] = 1
		eval_config = {
			"data_name": data_name,
			"eval_method": eval_method,
			"res_dir": result_dir,
			"test_data_file": test_data_file,
			"train_data_file": train_data_file,
			"e2e_fname": e2e_fname,
			"n_fixed_anc_ent": nm_train,
			"n_seeds": n_seeds,
		
			"misc": ("d=train" if eval_on_train_data else "d=test")
			
			
			# "bi_model_file": bi_model_file, :Not needed for cur eval
			# "batch_size": batch_size, :Not needed for cur eval
		}
		
		all_eval_configs += [eval_config]
		
	return all_eval_configs, all_train_configs, all_train_param_configs


def get_tfidf_eval_job_configs(eval_method, base_data_dir, base_res_dir, eval_on_train_data):
	

	
	n_seeds = 1
	all_train_configs, all_train_param_configs = _get_param_config()
	
	all_eval_configs = []
	processed_res_dirs = {}
	for ctr, curr_config in enumerate(all_train_configs):

		data_name = all_train_param_configs[ctr]['data_name']
		nm_train = all_train_param_configs[ctr]['num_train']
		split_idx = all_train_param_configs[ctr]['split_num']
		# misc = all_train_param_configs[ctr]['misc']
		
		train_data_file = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train.pkl"
		if eval_on_train_data:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train_train.pkl"
		else:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/test.pkl"
		
	
		train_worlds =  ["american_football", "doctor_who", "fallout", "final_fantasy", "military", "pro_wrestling", "starwars", "world_of_warcraft"]
		test_worlds = ["forgotten_realms", "lego", "star_trek", "yugioh"]
		valid_worlds = ["coronation_street", "elder_scrolls", "ice_hockey", "muppets"]
		
		if data_name in train_worlds:
			world_type = "train"
		elif data_name in test_worlds:
			world_type = "test"
		elif data_name in valid_worlds:
			world_type = "valid"
		else:
			raise NotImplementedError(f"World type for data_name = {data_name} not known")
	
		data_dir = "../../data/zeshel"
		mention_file =  f"{data_dir}/processed/{world_type}_worlds/{data_name}_mentions.jsonl"
		entity_file =  f"{data_dir}/documents/{data_name}.json"
		
		result_dir = f"{base_res_dir}/models/{data_name}/nm_train={nm_train}/split_idx={split_idx}/{eval_method}_s={n_seeds}"
		
		# There are multiple train configs that train on the exact same data so avoid running cur eval multiple times on the same test data
		if result_dir in processed_res_dirs: continue
		
		processed_res_dirs[result_dir] = 1
		eval_config = {
			"data_name": data_name,
			"eval_method": eval_method,
			"res_dir": result_dir,
			"test_data_file": test_data_file,
			"train_data_file": train_data_file,
			
			"mention_file": mention_file,
			"entity_file": entity_file,
		
			"misc": ("d=train" if eval_on_train_data else "d=test")
		}
		
		all_eval_configs += [eval_config]
		
	return all_eval_configs, all_train_configs, all_train_param_configs


def get_standard_bienc_eval_job_configs(base_data_dir, base_res_dir, eval_on_train_data):
	
	batch_size = 50
	all_train_configs, all_train_param_configs = _get_param_config()
	
	all_eval_configs = []
	processed_res_dirs = {}
	for ctr, curr_config in enumerate(all_train_configs):

		data_name = all_train_param_configs[ctr]['data_name']
		nm_train = all_train_param_configs[ctr]['num_train']
		split_idx = all_train_param_configs[ctr]['split_num']
		# misc = all_train_param_configs[ctr]['misc']
		
		train_data_file = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train.pkl"
		if eval_on_train_data:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/train_train.pkl"
		else:
			test_data_file  = f"{base_data_dir}/{data_name}/m2e_splits/nm_train={nm_train}/split_idx={split_idx}/test.pkl"
		
		
		bi_model_file = "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt"
		result_dir      = f"{base_res_dir}/models/{data_name}/nm_train={nm_train}/split_idx={split_idx}/bienc_6_20"
		
		# There are multiple train configs that train on the exact same data so avoid running cur eval multiple times on the same test data
		if result_dir in processed_res_dirs: continue
		
		processed_res_dirs[result_dir] = 1
		eval_config = {
			"data_name": data_name,
			"eval_method": "bienc",
			"res_dir": result_dir,
			"test_data_file": test_data_file,
			"train_data_file": train_data_file,
		
			"bi_model_file": bi_model_file,
			"batch_size": batch_size,
			
			"misc": "d=train" if eval_on_train_data else "d=test",
			
			# "n_seeds": n_seeds, Not needed for bienc eval
		}
		
		all_eval_configs += [eval_config]
		
	return all_eval_configs, all_train_configs, all_train_param_configs




def launch_eval_jobs():
	arg_eval_method = sys.argv[1]
	launch_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else False
	allow_running_even_if_already_run = int(sys.argv[3]) if len(sys.argv) > 3 else False
	eval_on_train_data = int(sys.argv[4]) if len(sys.argv) > 4 else False
	print_command = int(sys.argv[5]) if len(sys.argv) > 5 else False
	
	base_data_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/6_256_e_crossenc/score_mats_model-2-15999.0--79.46.ckpt"
	base_res_dir  = "../../results/8_CUR_EMNLP/d=ent_link_ce"
	

	if arg_eval_method == "cur":
		eval_method = arg_eval_method
		all_eval_configs, _, _ = get_cur_eval_job_configs(base_data_dir=base_data_dir, base_res_dir=base_res_dir, eval_on_train_data=eval_on_train_data)
	elif arg_eval_method in ["tfidf"]:
		eval_method = arg_eval_method
		all_eval_configs, _, _ = get_tfidf_eval_job_configs(eval_method=arg_eval_method, base_data_dir=base_data_dir, base_res_dir=base_res_dir, eval_on_train_data=eval_on_train_data)
	elif arg_eval_method in ["fixed_anc_ent", "fixed_anc_ent_cur"]:
		eval_method = arg_eval_method
		all_eval_configs, _, _ = get_fixed_anc_ent_eval_job_configs(base_data_dir=base_data_dir, base_res_dir=base_res_dir, eval_on_train_data=eval_on_train_data, eval_method=eval_method)
	elif arg_eval_method in ["bienc~eoe", "bienc~best_wrt_dev"]:
		ckpt_type = arg_eval_method.split("~")[-1]
		eval_method = arg_eval_method.split('~')[0]
		all_eval_configs, _, _ = get_bienc_eval_job_configs(base_data_dir=base_data_dir, ckpt_type=ckpt_type, eval_on_train_data=eval_on_train_data)
	elif arg_eval_method == "bienc_6_20":
		eval_method = "bienc"
		all_eval_configs, _, _ = get_standard_bienc_eval_job_configs(base_data_dir=base_data_dir, base_res_dir=base_res_dir, eval_on_train_data=eval_on_train_data)
	else:
		raise NotImplementedError(f"eval method = {arg_eval_method} not supported")
	
	
	all_commands = []
	previously_run_commands = []
	found = 0
	for ctr, curr_eval_config in enumerate(all_eval_configs):
		# curr_command = f"sbatch -p 2080ti-long --gres gpu:1 --mem 32GB --job-name " \
		# curr_command = f"sbatch -p gypsum-titanx-phd --mem 128GB --job-name " \
		# curr_command = f"sbatch -p cpu --mem 64GB --time 336:00:00 --job-name " \
		curr_command = f"sbatch -p gpu --gres gpu:1 --mem 32GB --time 336:00:00 --exclude gpu-0-0 --job-name " \
					   f" eval_{eval_method}_{curr_eval_config['data_name']}_{ctr} bin/run.sh python " \
					   f" eval/run_emnlp_retrieval_eval_wrt_exact_crossenc.py "
		
		
		for key, val in curr_eval_config.items():
			if val != "":
				curr_command += f" --{key} {val} "
		
		curr_command += " --use_wandb 1 "
		curr_command += " --mode eval "
		
		
		res_file = f"{curr_eval_config['res_dir']}/method={eval_method}_{curr_eval_config['misc']}.json"
		
		if allow_running_even_if_already_run or not os.path.isfile(res_file):
			all_commands += [curr_command]
			if launch_jobs:
				os.system(command=curr_command)
				time.sleep(1)
			if print_command: print(f"{curr_command}\n")
			# LOGGER.info(f"curr_res_dir: {result_dir}")
		else:
			previously_run_commands += [curr_command]
	
	
	LOGGER.info(f"Previously run commands = {len(previously_run_commands)}")
	LOGGER.info(f"Commands run now = {len(all_commands)}")
	LOGGER.info(f"Found = {found}")


if __name__ == "__main__":
	# launch_train_jobs()
	launch_eval_jobs()
	pass
