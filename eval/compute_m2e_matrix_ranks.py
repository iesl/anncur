import os
import sys
import torch
import pickle
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path

from eval.eval_utils import compute_label_embeddings, compute_input_embeddings
from models.biencoder import BiEncoderWrapper
from utils.zeshel_utils import N_ENTS_ZESHEL, N_MENTS_ZESHEL
from utils.data_process import load_entities, load_mentions, get_context_representation
from utils.zeshel_utils import get_dataset_info, get_zeshel_world_info, MAX_ENT_LENGTH, MAX_MENT_LENGTH


logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)




def main():
	
	data_name = "yugioh"
	dir_list = {
		 "Random-ECE": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=random_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-3-23399.0--98.07.ckpt",
		 "Random-ClS-CE": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=random_s=1234_63_negs_w_cls_w_lin/score_mats_model-3-23399.0--98.18.ckpt",
		
		 "TFIDF-E-CE": "../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_crossenc_w_embeds_tfidf_hard_negs/score_mats_model-1-11079.0--91.17.ckpt",
		 "TFIDF-ClS-CE": "../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_cls_w_lin_tfidf_hard_negs/score_mats_model-1-12279.0--90.95.ckpt",
		
		 "BIENC-E-CE": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt",
		 "BIENC-ClS-CE": "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt",
	}
	for curr_dir_type, curr_dir in dir_list.items():
		
		curr_file = f"{curr_dir}/{data_name}/ment_to_ent_scores_n_m_{N_MENTS_ZESHEL[data_name]}_n_e_{N_ENTS_ZESHEL[data_name]}_all_layers_False.pkl"
		try:
			with open(curr_file, "rb") as fin:
				dump_dict = pickle.load(fin)
				ment_to_ent_scores = dump_dict["ment_to_ent_scores"].numpy()
				
				rank = np.linalg.matrix_rank(ment_to_ent_scores)
				LOGGER.info(f"Curr_dir type:{curr_dir_type}")
				LOGGER.info(f"Shape of matrix = {ment_to_ent_scores.shape}")
				LOGGER.info(f"Rank of matrix = {rank}\n\n")
		except Exception as e:
			LOGGER.info(f"File not found for {curr_dir_type}\n")
			
		
def compute_binec_ment_to_ent_scores():
	
	data_dir = "../../data/zeshel"
	data_name = "yugioh"
	worlds = get_zeshel_world_info()

	DATASETS = get_dataset_info(data_dir=data_dir, worlds=worlds, res_dir=None)
	data_fname = DATASETS[data_name]
	
	bi_model_file = "../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt"
	out_file = f"../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/score_mats_model-3-12039.0-2.17.ckpt/{data_name}/ment_to_ent_scores_{3374}x{10031}.pkl"
	
	
	biencoder = BiEncoderWrapper.load_from_checkpoint(bi_model_file)
	biencoder.eval()
	
	
	(title2id,
	id2title,
	id2text,
	kb_id2local_id) = load_entities(entity_file=data_fname["ent_file"])
	
	tokenizer = biencoder.tokenizer

	test_data = load_mentions(mention_file=data_fname["ment_file"], kb_id2local_id=kb_id2local_id)
	
	# First extract all mentions and tokenize them
	mention_tokens_list = [get_context_representation(sample=mention, tokenizer=tokenizer, max_seq_length=MAX_MENT_LENGTH)["ids"]
							for mention in tqdm(test_data)]
	
	curr_mentions_tensor = torch.LongTensor(mention_tokens_list)
	complete_entity_tokens_list = torch.LongTensor(np.load(data_fname["ent_tokens_file"]))
	
	entity_embeds = compute_label_embeddings(
		biencoder=biencoder,
		labels_tokens_list=complete_entity_tokens_list,
		batch_size=50
	)
	
	
	mention_embeds = compute_input_embeddings(
		biencoder=biencoder,
		input_tokens_list=curr_mentions_tensor,
		batch_size=50
	)
	
	
	ment_to_ent_matrix = mention_embeds @ entity_embeds.T
	
	dump_dict =  {
		"ment_to_ent_scores": ment_to_ent_matrix
	}
	
	Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
	with open(out_file, "wb") as fout:
		
		pickle.dump(dump_dict, fout)


if __name__ == "__main__":

	# compute_binec_ment_to_ent_scores()
	main()
	pass
