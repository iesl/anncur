import os
import sys
import json
import time
import argparse

from tqdm import tqdm
import logging
import itertools
import numpy as np
from IPython import embed
from pathlib import Path
import pickle
from utils.zeshel_utils import N_ENTS_ZESHEL, N_MENTS_ZESHEL

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _save_data_for_given_ments(split_name, out_dir, ment_idxs, ment_to_ent_scores, mention_data, mention_tokens_list, arg_dict, ):
	
	try:
		if len(ment_idxs) == 0:
			LOGGER.info(f"Empty list of indices for split = {split_name}")
			return
		curr_ment_to_ent_scores = ment_to_ent_scores[ment_idxs, :]
		curr_mention_data = [mention_data[idx] for idx in ment_idxs]
		curr_mention_tokens_list = [mention_tokens_list[idx] for idx in ment_idxs]
	
		curr_m2e_dict = {
			"ment_to_ent_scores": curr_ment_to_ent_scores,
			"test_data": curr_mention_data, # This test_data key is used in zeshel entity linking exps but this does not correspond to test data in this setting.
			"mention_tokens_list": curr_mention_tokens_list,
			"ment_idxs": ment_idxs,
			"entity_id_list": [],
			"entity_tokens_list": [], # No need of saving this as long as all entities as scored
			"arg_dict": arg_dict
		}
		
		Path(out_dir).mkdir(exist_ok=True, parents=True)
		out_file = f"{out_dir}/{split_name}.pkl"
		with open(out_file, "wb") as fout:
			pickle.dump(curr_m2e_dict, fout)
	except Exception as e:
		embed()
		raise e


def run(data_name, m2e_file, num_train_ment_vals, num_splits, seed, dev_frac, base_out_dir):
	
	rng = np.random.default_rng(seed=seed)
	assert 0 <= dev_frac < 1
	with open(m2e_file, "rb") as fin:
		dump_dict = pickle.load(fin)
		
		ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
		mention_data = dump_dict["test_data"]
		mention_tokens_list = dump_dict["mention_tokens_list"]
		entity_id_list = dump_dict["entity_id_list"]
		# entity_tokens_list = dump_dict["entity_tokens_list"]
		arg_dict = dump_dict["arg_dict"]
	
		# This is needed as we don't save entity_id_list for each split. Is this condition is not true then we should!
		assert len(entity_id_list) == 0 or (entity_id_list == np.arange(N_ENTS_ZESHEL[data_name])).all()
		
		
		n_ments = ment_to_ent_scores.shape[0]
		
		assert n_ments == len(mention_data)
		assert n_ments == len(mention_tokens_list)
		
		
		for num_train_ments, split_iter in tqdm(itertools.product(num_train_ment_vals, range(num_splits))):
			if num_train_ments > n_ments:
				LOGGER.info(f"Number of train ments = {num_train_ments} > n_ments = {n_ments}")
				continue
				
			# Split data
			train_ment_idxs = sorted(rng.choice(n_ments, size=num_train_ments, replace=False))
			test_ment_idxs  = sorted(list(set(list(range(n_ments))) - set(train_ment_idxs)))
			
			# Now split train into train-and-dev
			train_dev_ment_idxs = sorted(rng.choice(a=train_ment_idxs, size=int(num_train_ments*dev_frac), replace=False))
			train_train_ment_idxs = sorted(list(set(train_ment_idxs) - set(train_dev_ment_idxs)))
			
			_save_data_for_given_ments(
				ment_idxs=train_dev_ment_idxs,
				split_name="train_dev",
				ment_to_ent_scores=ment_to_ent_scores,
				mention_data=mention_data,
				mention_tokens_list=mention_tokens_list,
				arg_dict=arg_dict,
				out_dir=f"{base_out_dir}/nm_train={num_train_ments}/split_idx={split_iter}"
			)
			
			_save_data_for_given_ments(
				ment_idxs=train_train_ment_idxs,
				split_name="train_train",
				ment_to_ent_scores=ment_to_ent_scores,
				mention_data=mention_data,
				mention_tokens_list=mention_tokens_list,
				arg_dict=arg_dict,
				out_dir=f"{base_out_dir}/nm_train={num_train_ments}/split_idx={split_iter}"
			)
			
			_save_data_for_given_ments(
				ment_idxs=train_ment_idxs,
				split_name="train",
				ment_to_ent_scores=ment_to_ent_scores,
				mention_data=mention_data,
				mention_tokens_list=mention_tokens_list,
				arg_dict=arg_dict,
				out_dir=f"{base_out_dir}/nm_train={num_train_ments}/split_idx={split_iter}"
			)
			
			_save_data_for_given_ments(
				ment_idxs=test_ment_idxs,
				split_name="test",
				ment_to_ent_scores=ment_to_ent_scores,
				mention_data=mention_data,
				mention_tokens_list=mention_tokens_list,
				arg_dict=arg_dict,
				out_dir=f"{base_out_dir}/nm_train={num_train_ments}/split_idx={split_iter}"
			)
	

def main():

	parser = argparse.ArgumentParser( description='Split zeshel mention-entity score matrices into train/test mentions')
	
	parser.add_argument("--data_name", type=str, required=True, help="Data/domain name")
	parser.add_argument("--m2e_file", type=str, required=True, help="Mention-Entity score file")
	parser.add_argument("--out_dir", type=str, default="", help="Output dir")
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	parser.add_argument("--dev_frac", type=float, default=0.1, help="Random seed")
	parser.add_argument("--num_splits", type=int, default=5, help="Number of random splits")
	
	args = parser.parse_args()
	
	data_name = args.data_name
	m2e_file = args.m2e_file
	num_splits = args.num_splits
	seed = args.seed
	dev_frac = args.dev_frac
	
	base_out_dir = os.path.dirname(m2e_file) if args.out_dir == "" else args.out_dir
	num_train_ment_vals = [50, 100, 200, 500, 1000, 2000]
	
	
	out_dir = f"{base_out_dir}/m2e_splits"
	run(
		data_name=data_name,
		m2e_file=m2e_file,
		num_train_ment_vals=num_train_ment_vals,
		num_splits=num_splits,
		seed=seed,
		base_out_dir=out_dir,
		dev_frac=dev_frac
	)

	# Save some meta info about how this split was created
	with open(f"{out_dir}/split_args.json", "w") as fout:
		json.dump(args.__dict__, fout)

	

if __name__ == "__main__":
	main()
