import sys
import math
import json
import torch
import faiss
import logging
import argparse

import numpy as np
from tqdm import tqdm
from IPython import embed

from eval.eval_utils import compute_label_embeddings

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def build_flat_or_ivff_index(embeds, force_exact_search, probe_mult_factor=1):
	
	LOGGER.info(f"Beginning indexing given {len(embeds)} embeddings")
	if type(embeds) is not np.ndarray:
		if torch.is_tensor(embeds):
			embeds = embeds.numpy()
		else:
			embeds = np.array(embeds)
	
	# Build index
	d = embeds.shape[1] # Dimension of vectors to embed
	nembeds = embeds.shape[0] # Number of elements to embed
	if nembeds <= 11000 or force_exact_search:  # if the number of embeddings is small, don't approximate
		index = faiss.IndexFlatIP(d)
		index.add(embeds)
	else:
		# number of quantized cells
		nlist = int(math.floor(math.sqrt(nembeds)))
		
		# number of the quantized cells to probe
		nprobe = int(math.floor(math.sqrt(nlist) * probe_mult_factor))
		
		quantizer = faiss.IndexFlatIP(d)
		index = faiss.IndexIVFFlat(
			quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
		)
		index.train(embeds)
		index.add(embeds)
		index.nprobe = nprobe
	
	LOGGER.info("Finished indexing given embeddings")
	return index


def embed_tokenized_entities(biencoder, ent_tokens_file):
	
	complete_entity_tokens_list = np.load(ent_tokens_file)
	
	# complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list).to(biencoder.device)
	complete_entity_tokens_list = torch.LongTensor(complete_entity_tokens_list)
	
	embeds = compute_label_embeddings(biencoder=biencoder,
									  labels_tokens_list=complete_entity_tokens_list,
									  batch_size=50)
	
	return embeds


def index_tokenized_entities(biencoder, ent_tokens_file, force_exact_search):
	
	embeds = embed_tokenized_entities(biencoder=biencoder,
							 ent_tokens_file=ent_tokens_file)
	
	index = build_flat_or_ivff_index(embeds=embeds,
									 force_exact_search=force_exact_search)
	
	return index, embeds


def temp_search(embeds, index):
	try:
		k = 10
		for idx, curr_embed in enumerate(tqdm(embeds)):
			curr_embed = curr_embed.reshape(1,-1)
			nn_ent_dists, nn_ent_idxs = index.search(curr_embed, k)
			embed()
			input("")
			# if idx > 10:
			# 	break
	except Exception as e:
		embed()
		raise e


def main():
	
	parser = argparse.ArgumentParser( description='Build Nearest Nbr Index')
	
	parser.add_argument("--ent_file", type=str, required=True, help="File containing tokenized entities or entity embeddings")
	# parser.add_argument("--out_file", type=str, required=True, help="Output file storing index")
	
	parser.add_argument("--model_config", type=str, default=None, help="Model config for embedding model")
	
	
	args = parser.parse_args()
	
	ent_file = args.ent_file
	model_config = args.model_config
	
	
	from models.biencoder import BiEncoderWrapper
	if model_config is not None:
		with open(model_config, "r") as fin:
			config = json.load(fin)
			biencoder = BiEncoderWrapper.load_model(config=config)
			
		index, embeds = index_tokenized_entities(biencoder=biencoder,
								 ent_tokens_file=ent_file,
								 force_exact_search=False)
	else:
		embeds = np.load(ent_file)
		index = build_flat_or_ivff_index(embeds=embeds,
										 force_exact_search=False)
		
	temp_search(embeds=embeds, index=index)
	


if __name__ == "__main__":
	main()
