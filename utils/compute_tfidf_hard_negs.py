import os
import sys
import json
import logging
import argparse

from IPython import embed
from pathlib import Path
import numpy as np

from utils.zeshel_utils import get_dataset_info
from utils.zeshel_utils import get_zeshel_world_info
from utils.data_process import get_hard_negs_tfidf, read_ent_link_data
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def compute_negs(data_name, base_res_dir, num_negs, mention_file, entity_file):
	
	try:
		LOGGER.info(f"Reading data from {mention_file}, {entity_file}")
		mention_data, entity_data  = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
		
		pos_label_idxs = [[int(mention["label_id"])] for mention in mention_data]
		
		LOGGER.info("Finding hard tfidf negatives")
		neg_labels_idxs = get_hard_negs_tfidf(
			mentions_data=mention_data,
			entity_file=entity_file,
			pos_label_idxs=pos_label_idxs,
			num_negs=num_negs,
			force_exact_search=True
		)
		
		LOGGER.info("Done finding hard negatives")
		
		out_file = f"{base_res_dir}/{data_name}/tfidf_hard_negs_n={num_negs}.json"
		
		neg_data = {
			"indices": neg_labels_idxs.tolist(),
			"scores": np.ones(neg_labels_idxs.shape).tolist()
		}
		
		
		LOGGER.info(f"Saving data in file = {out_file}")
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		with open(out_file, "w") as fout:
			json.dump(neg_data, fout)
	except Exception as e:
		embed()
		raise e



def main():

	worlds = get_zeshel_world_info()

	parser = argparse.ArgumentParser(description='Compute TF-IDF hard negatives')
	parser.add_argument("--res_dir", type=str, required=True, help="Result directory")
	parser.add_argument("--data_dir", type=str, required=True, help="Directory containing ZeShEL data. Usually this would = ../../data/zeshel")
	parser.add_argument("--data_name", type=str, required=True, choices=[w for _,w in worlds], help="Domain name in ZeShEL")
	parser.add_argument("--num_negs", type=int, required=True, help="Number of negatives")
	
	args = parser.parse_args()
	data_name = args.data_name
	num_negs = args.num_negs
	base_res_dir = args.res_dir
	data_dir = args.data_dir
	
	worlds = get_zeshel_world_info()
	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	
	compute_negs(
		data_name=data_name,
		base_res_dir=base_res_dir,
		entity_file=DATASETS[data_name]["ent_file"],
		mention_file=DATASETS[data_name]["ment_file"],
		num_negs=num_negs
	)
	
	

if __name__ == "__main__":
	main()

