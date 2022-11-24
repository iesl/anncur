import sys
import json
import argparse

from tqdm import tqdm
import logging
import torch
import numpy as np
from IPython import embed
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from utils.data_process import load_entities, load_mentions, get_context_representation
from eval.eval_utils import score_topk_preds, compute_label_embeddings
from utils.zeshel_utils import get_zeshel_world_info, get_dataset_info, MAX_MENT_LENGTH, MAX_ENT_LENGTH, MAX_PAIR_LENGTH
from models.biencoder import BiEncoderWrapper

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)



def run(biencoder, data_fname, n_ment, batch_size, top_k, res_dir, dataset_name, misc):
	try:
		assert top_k > 1
		rng = np.random.default_rng(seed=0)
		
		biencoder.eval()
		(title2id,
		id2title,
		id2text,
		kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
		
		tokenizer = biencoder.tokenizer
		max_ent_length = MAX_ENT_LENGTH
		max_ment_length = MAX_MENT_LENGTH
		max_pair_length = MAX_PAIR_LENGTH
		
		test_data = load_mentions(mention_file=data_fname["ment_file"],
								  kb_id2local_id=kb_id2local_id)
		
		test_data = test_data[:n_ment] if n_ment > 0 else test_data
		# First extract all mentions and tokenize them
		mention_tokens_list = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=max_ment_length)["ids"]
								for mention in tqdm(test_data)]
		
		curr_mentions_tensor = torch.LongTensor(mention_tokens_list)
		curr_gt_labels = np.array([x["label_id"] for x in test_data])
		
		batched_data = TensorDataset(curr_mentions_tensor)
		bienc_dataloader = DataLoader(batched_data, batch_size=batch_size, shuffle=False)
		
		complete_entity_tokens_list = np.load(data_fname["ent_tokens_file"])
		complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
		
		candidate_encoding = compute_label_embeddings(biencoder=biencoder, labels_tokens_list=complete_entity_tokens_list,
													  batch_size=batch_size)
		# # candidate_encoding = torch.Tensor(np.load(data_fname["ent_embed_file"]))
		candidate_encoding = candidate_encoding.t() # Take transpose for easier matrix multiplication ops later
		
	
		bienc_topk_preds = []
		with torch.no_grad():
			biencoder.eval()
			torch.cuda.empty_cache()
			LOGGER.info(f"Starting computation with batch_size={batch_size}, n_ment={n_ment}, top_k={top_k}")
			LOGGER.info(f"Bi encoder model device {biencoder.device}")
			for batch_idx, (batch_ment_tokens,) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
				
				batch_ment_tokens =  batch_ment_tokens.to(biencoder.device)
				
				ment_encodings = biencoder.encode_input(batch_ment_tokens)
				ment_encodings = ment_encodings.to(candidate_encoding.device)
				batch_bienc_scores = ment_encodings.mm(candidate_encoding)
				
				# Use batch_idx here as anc_ment_to_ent_scores only contain scores for anchor mentions.
				# If it were complete mention-entity matrix then we would have to use ment_idx
				batch_bienc_top_k_scores, batch_bienc_top_k_indices = batch_bienc_scores.topk(top_k)
				
				bienc_topk_preds += [(batch_bienc_top_k_indices, batch_bienc_top_k_scores)]
				torch.cuda.empty_cache()
				
		
		curr_res_dir = f"{res_dir}/{dataset_name}/m={n_ment}_k={top_k}_{batch_size}_{misc}"
		Path(curr_res_dir).mkdir(exist_ok=True, parents=True)
		
		bienc_topk_preds = _get_indices_scores(bienc_topk_preds)
		
		json.dump(curr_gt_labels.tolist(), open(f"{curr_res_dir}/gt_labels.txt", "w"))
		json.dump(bienc_topk_preds, open(f"{curr_res_dir}/bienc_topk_preds.txt", "w"))
		
		with open(f"{curr_res_dir}/res.json", "w") as fout:
			res = {"bienc": score_topk_preds(gt_labels=curr_gt_labels,
											 topk_preds=bienc_topk_preds)}
			json.dump(res, fout, indent=4)
			LOGGER.info(json.dumps(res, indent=4))
		
		
		LOGGER.info("Done")
	except Exception as e:
		LOGGER.info(f"Error raised {str(e)}")
		embed()



def _get_indices_scores(topk_preds):
	indices, scores = zip(*topk_preds)
	indices, scores = torch.cat(indices), torch.cat(scores)
	indices, scores = indices.cpu().numpy().tolist(), scores.cpu().numpy().tolist()
	return {"indices":indices, "scores":scores}




def main():
	worlds = get_zeshel_world_info()
	parser = argparse.ArgumentParser( description='Run biencoder model on given mention data')
	parser.add_argument("--data_name", type=str, choices=[w for _,w in worlds] + ["all"], help="Dataset name")
	parser.add_argument("--n_ment", type=int, default=-1, help="Number of mentions. -1 for all mentions")
	parser.add_argument("--top_k", type=int, default=100, help="Top-k mentions to retrieve using bi-encoder")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	
	parser.add_argument("--model_config", type=str, required=False, default="", help="Model config file")
	parser.add_argument("--model_ckpt", type=str, required=False, default="", help="Model ckpt file")
	parser.add_argument("--data_dir", type=str, default="../../data/zeshel", help="Data dir")
	parser.add_argument("--res_dir", type=str, required=True, help="Dir to save results")
	parser.add_argument("--misc", type=str, default="", help="suffix for files/dir where results are saved")

	get_zeshel_world_info()
	args = parser.parse_args()

	data_dir = args.data_dir
	data_name = args.data_name
	n_ment = args.n_ment
	top_k = args.top_k
	batch_size = args.batch_size
	
	model_config = args.model_config
	model_ckpt = args.model_ckpt
	assert model_config != "" or model_ckpt != ""
	assert model_config == "" or model_ckpt == ""
	
	res_dir = args.res_dir
	misc = args.misc
	
	Path(res_dir).mkdir(exist_ok=True, parents=True)
	
	if model_config != "":
		with open(model_config, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
	else:
		biencoder = BiEncoderWrapper.load_from_checkpoint(model_ckpt)
	

	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
		
	iter_worlds = worlds if data_name == "all" else [("dummy", data_name)]
	
	for world_type, world_name in tqdm(iter_worlds):
		LOGGER.info(f"Running inference for world = {world_name}")
		run(biencoder=biencoder,  data_fname=DATASETS[world_name],
			n_ment=n_ment, top_k=top_k, batch_size=batch_size, dataset_name=data_name,
			res_dir=res_dir, misc=misc)



if __name__ == "__main__":
	main()

