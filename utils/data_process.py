import os
import sys
import json
import torch
import logging
import warnings
import pickle
import numpy as np

from tqdm import tqdm
from IPython import embed
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from pytorch_transformers.tokenization_bert import BertTokenizer

from models.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from eval.eval_utils import compute_label_embeddings, compute_input_embeddings
from models.nearest_nbr import build_flat_or_ivff_index
from utils.config import Config

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)



def load_raw_data(config, data_split_type):
	assert isinstance(config, Config)
	if config.data_type in ["ent_link"]:
		if data_split_type == "train":
			input_files = config.trn_files
		elif data_split_type == "dev":
			input_files = config.dev_files
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		all_data = {}
		for domain, (mention_file, entity_file, _) in input_files.items():
			mention_data, entity_data = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
			all_data[domain] = (mention_data, entity_data)
			
		return all_data
	elif config.data_type == "ent_link_ce":
		
		if data_split_type == "train":
			domains = config.train_domains
		elif data_split_type == "dev":
			domains = config.dev_domains
		else:
			raise Exception(f"data_type = {data_split_type} not supported")
		
		all_data = {}
		for domain in domains:
			mention_file = config.mention_file_template.format(domain)
			entity_file = config.entity_file_template.format(domain)
			mention_data, entity_data = read_ent_link_data(mention_file=mention_file, entity_file=entity_file)
			all_data[f"{data_split_type}~{domain}"] = (mention_data, entity_data)
			
		return all_data
	
	else:
		raise Exception(f"Data type = {config.data_type} is not supported")


def read_ent_link_data(mention_file, entity_file):
	"""
	Load mention and entity data for entity linking
	:param mention_file:
	:param entity_file:
	:return:
	"""
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=entity_file)
	
	mention_data = load_mentions(mention_file=mention_file, kb_id2local_id=kb_id2local_id)
	
	return mention_data, (title2id, id2title, id2text, kb_id2local_id)


def load_mentions(mention_file, kb_id2local_id):
	"""
	Load mention data
	:param mention_file: Each line contains data about
	:param kb_id2local_id: Dict mapping KB entity id to local entity id. Mention file contains
							 KB id for ground-truth entities so we need to map
							 those KB entity ids to local entity ids
	
	:return: List of mentions.
			Each mention is a dict with four keys :  label_id, context_left, context_right, and mention
	"""
	assert kb_id2local_id and len(kb_id2local_id) > 0, f"kb_id2local_id = {kb_id2local_id} is empty!!!"
	
	test_samples = []
	with open(mention_file, "r") as fin:
		lines = fin.readlines()
		for line in lines:
			record = json.loads(line)
			label_id = record["label_id"]
			
			# check that each entity id (label_id) is in the entity collection
			if label_id not in kb_id2local_id:
				continue
			
			# LOWERCASE EVERYTHING !
			new_record = {"label_id": kb_id2local_id[label_id],
						  "context_left": record["context_left"].lower(),
						  "context_right": record["context_right"].lower(),
						  "mention": record["mention"].lower()
						  }
			test_samples.append(new_record)
	
	LOGGER.info("{}/{} samples considered".format(len(test_samples), len(lines)))
	return test_samples


def load_entities(entity_file):
	"""
	Load entity data
	:param entity_file: File containing entity data. Each line entity info as a json.
	:return: Dicts
		title2id, id2title, id2text, kb_id2local_id
		kb_id2local_id -> Maps KB entity id to local entity id
	"""
	
	# load all entities in entity_file
	title2id = {}
	id2title = {}
	id2text = {}
	kb_id2local_id = {}
	
	with open(entity_file, "r") as fin:
		lines = fin.readlines()
		for local_idx, line in enumerate(lines):
			entity = json.loads(line)
			
			if "idx" in entity: # For Wikipedia entities
				split = entity["idx"].split("curid=")
				if len(split) > 1:
					kb_id = int(split[-1].strip())
				else:
					kb_id = entity["idx"].strip()
			else: # For ZeShEL entities
				kb_id = entity["document_id"]
			
			
			assert kb_id not in kb_id2local_id
			kb_id2local_id[kb_id] = local_idx
			
			
			title2id[entity["title"]] = local_idx
			id2title[local_idx] = entity["title"]
			id2text[local_idx] = entity["text"]
	
	return (
		title2id,
		id2title,
		id2text,
		kb_id2local_id,
	)


def compute_ment_embeds_w_tfidf(entity_file, mentions):
	"""
	Trains a tfidf vectorizer using entity file and then vectorizes mentions using trained tfidf vectorizer
	:param entity_file: File containing entity information
	:param mentions: List of mention strings.
	:return: TF_IDF embeddings of mentions
	"""
	LOGGER.info("\t\tLoading entity descriptions")
	# Read entity descriptions and embed using tf-idf
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=entity_file)
	
	LOGGER.info("\t\tTraining vectorizer on entity descriptions")
	### Build a list of entity description and train a vectorizer
	corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
	vectorizer = TfidfVectorizer(dtype=np.float32)
	vectorizer.fit(corpus)
	
	### Embed all entities usign tfidf vectorizer
	LOGGER.info("\t\tTransforming mentions to sparse vectors")
	
	ment_embeds = vectorizer.transform(mentions)
	
	return np.asarray(ment_embeds.todense())


def compute_ment_embeds_w_bienc(biencoder, mention_tokens_list):
	"""
	Embed mentions using biencoder model
	:param biencoder: Biencoder model for embedding mentions
	:param mention_tokens_list: List of tokenized mentions
	:return: torch tensor containing mention embeddings
	"""
	with torch.no_grad():
		assert not biencoder.training, "Biencoder should be in eval mode"
		torch.cuda.empty_cache()
		bienc_ment_embedding = []
		all_mention_tokens_list_gpu = torch.tensor(mention_tokens_list).to(biencoder.device)
		for ment in all_mention_tokens_list_gpu:
			ment = ment.unsqueeze(0)
			bienc_ment_embedding += [biencoder.encode_input(ment)]
		
		bienc_ment_embedding = torch.cat(bienc_ment_embedding)
	
	return bienc_ment_embedding


def compute_ment_embeds(embed_type, entity_file, mentions, biencoder, mention_tokens_list):
	"""
	Computes  mention embeddings with given method
	:param embed_type: Method to use for computing mention embeddings
	:param entity_file: File containing entity information
	:param mentions: List of mention strings.
	:param biencoder: Biencoder model for embedding mentions
	:param mention_tokens_list: List of tokenized mentions
	:return: Array containing mention embeddings
	"""
	if embed_type == "tfidf":
		ment_embeds = compute_ment_embeds_w_tfidf(
			entity_file=entity_file,
			mentions=mentions
		)
		return ment_embeds
	elif embed_type == "bienc":
		LOGGER.info("\t\tComputing mention embedding using biencoder")
		ment_embeds = compute_ment_embeds_w_bienc(biencoder=biencoder, mention_tokens_list=mention_tokens_list)
		ment_embeds = ment_embeds.cpu().detach().numpy()
		return ment_embeds
	elif embed_type == "random":
		return None
	else:
		raise NotImplementedError(f"Method = {embed_type} not supported for computing mention embeddings")


def compute_ent_embeds_w_tfidf(entity_file):
	"""
	Trains a tf-idf vectorizer over entity title and text, vectorizes them, and returns dense tfidf embeddings
	:param entity_file: File containing entity information
	:return:
	"""
	LOGGER.info("Loading entity descriptions")
	# Read entity descriptions and embed using tf-idf
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=entity_file)
	
	LOGGER.info("Training ")
	### Build a list of entity description and train a vectorizer
	corpus = [f"{id2title[curr_id]} {id2text[curr_id]}" for curr_id in sorted(id2title)]
	vectorizer = TfidfVectorizer(dtype=np.float32)
	vectorizer.fit(corpus)
	
	### Embed all entities usign tfidf vectorizer
	LOGGER.info("Transforming entity to sparse vectors")
	label_embeds = vectorizer.transform(corpus)
	
	return label_embeds.toarray()


def get_random_negs(data, n_labels, num_negs, seed, label_key):
	"""
	Sample random negative for each datapoint
	:param data: List of datapoints. Each datapoint is a dictionary storing info such as input text, label_idxs etc
	:param n_labels: Total number of labels
	:param num_negs:
	:param seed:
	:param label_key: Key in datapoint dictionary that stores labels for the corresponding datapoint
	:return:
	"""
	rng = np.random.default_rng(seed)
	
	neg_labels = []
	for datapoint in data:
		p = np.ones(n_labels)
		p[datapoint[label_key]] = 0  # Remove positive labels from list of allowed labels
		p = p / np.sum(p)
		neg_idxs = rng.choice(n_labels, size=num_negs, replace=False, p=p)
		
		# Add neg_labels for this datapoints as many times as the number of positive labels for it
		neg_labels += [neg_idxs] * len(datapoint[label_key]) if isinstance(datapoint[label_key], list) else [neg_idxs]
	
	return neg_labels


def get_random_negs_w_blacklist(n_data, n_labels, num_negs, label_blacklist, seed,):
	"""
	Sample random negative for each datapoint from all labels except for labels blacklisted for corresponding datapoint
	:param n_labels: Total number of labels
	:param num_negs:
	:param seed:
	:param label_blacklist: 2-D List containing labels to ignore for correspondign datapoints
	:return:
	"""
	rng = np.random.default_rng(seed)
	
	neg_labels = []
	for ctr in range(n_data):
		p = np.ones(n_labels)
		p[label_blacklist[ctr]] = 0  # Remove positive labels from list of allowed labels
		p = p / np.sum(p)
		neg_idxs = rng.choice(n_labels, size=num_negs, replace=False, p=p)
		
		neg_labels += [neg_idxs]
	
	return neg_labels


def get_hard_negs_biencoder(biencoder, input_tokens_list, labels_tokens_list, pos_label_idxs, num_negs):
	"""
	Embed inputs and labels, and then mine approx nearest nbr hard negatives for each input
	:param num_negs:
	:param pos_label_idxs:
	:param biencoder
	:param input_tokens_list:
	:param labels_tokens_list:
	:return:
	"""
	
	batch_size = 50

	# Embed tokenized labels and inputs
	label_embeds = compute_label_embeddings(biencoder=biencoder,
											labels_tokens_list=labels_tokens_list,
											batch_size=batch_size)
	
	input_embeds = compute_input_embeddings(biencoder=biencoder,
											input_tokens_list=input_tokens_list,
											batch_size=batch_size)
	
	# Build an index on labels
	nnbr_index = build_flat_or_ivff_index(embeds=label_embeds, force_exact_search=False)
	
	neg_labels = []
	neg_label_scores = []
	for curr_embed, curr_pos_labels in zip(input_embeds, pos_label_idxs):
		curr_pos_labels = set(curr_pos_labels)
		curr_embed = curr_embed.cpu().numpy()[np.newaxis, :]
		
		nn_scores, nn_idxs = nnbr_index.search(curr_embed, num_negs + len(curr_pos_labels))
		
		# Remove positive labels if there are present amongst nearest nbrs
		nn_idx_and_scores = [
							   (nn_idx, nn_score)
							   for nn_idx, nn_score in zip(nn_idxs[0], nn_scores[0])
							   if nn_idx not in curr_pos_labels
						   ][:num_negs]
		nn_idxs, nn_scores = zip(*nn_idx_and_scores)
		nn_idxs, nn_scores = list(nn_idxs), list(nn_scores)
		
		assert len(nn_idxs) == num_negs

		neg_labels += [nn_idxs]
		neg_label_scores += [nn_scores]
	
	neg_labels = np.array(neg_labels)
	neg_label_scores = np.array(neg_label_scores)
	
	return neg_labels, neg_label_scores


def get_hard_negs_tfidf(mentions_data, entity_file, pos_label_idxs, num_negs, force_exact_search=False):
	"""
	Compute hard negatives using tfidf embeddings of entities and mentions
	:return:
	"""
	############################# GET MENTION EMBEDDINGS USING TFIDF #########################
	n_ments = len(mentions_data)
	mentions = [" ".join([ment_dict["context_left"], ment_dict["mention"], ment_dict["context_right"]])
				for ment_dict in mentions_data]
	
	LOGGER.info(f"Embedding {n_ments} mentions using method = tfidf")
	ment_embeds = compute_ment_embeds(embed_type="tfidf", mentions=mentions, entity_file=entity_file,
									  mention_tokens_list=[], biencoder=None)
	
	LOGGER.info(f"Embedding entities using method = tfidf")
	
	# n_ents = len(ent_embeds)
	################################################################################################################
	
	LOGGER.info(f"Finding {num_negs}+1 nearest entities for {n_ments} mentions from all entities in file {entity_file}")
	nnbr_index = build_flat_or_ivff_index(
		embeds=compute_ent_embeds_w_tfidf(entity_file=entity_file),
		force_exact_search=force_exact_search
	)
	_, init_ents = nnbr_index.search(ment_embeds, num_negs + 1)
	
	# Remove positive labels if there are present amongst nearest nbrs
	final_init_ents = []
	for curr_init_ents, curr_pos_labels in tqdm(zip(init_ents, pos_label_idxs), total=len(pos_label_idxs)):
		curr_init_ents = [ent_idx for ent_idx in curr_init_ents if ent_idx not in curr_pos_labels][:num_negs]
		final_init_ents += [np.array(curr_init_ents)]
	
	final_init_ents = np.array(final_init_ents)
	
	return final_init_ents


def _sort_by_score(indices, scores):
	"""
	Sort each row in scores array in decreasing order and also permute each row of ent_indices accordingly
	:param indices: 2-D numpy array of indices
	:param scores: 2-D numpy array of scores
	:return:
	"""
	assert indices.shape == scores.shape, f"ent_indices.shape ={indices.shape}  != ent_scores.shape = {scores.shape}"
	n, m = scores.shape
	scores = torch.tensor(scores)
	topk_scores, topk_idxs = torch.topk(scores, m)
	sorted_ent_indices = np.array([indices[i][topk_idxs[i]] for i in range(n)])
	
	return sorted_ent_indices, topk_scores


def get_precomputed_ents_w_scores(ent_w_scores_file, n_ments, tokenized_entities, num_labels):
	"""
	Loads entities and entity scores associated from ent_w_scores_file.
	This can be used for mention and entity biencoder distillation or mention-entity crossencoder training.
	
	:param ent_w_scores_file: File containing entity indices and their scores to use for distillation
	:param n_ments: Number of ments
	:param tokenized_entities: Tensor containing tokenized entities
	:param num_labels: Number of entities to use per mention for purpose of distillation
	:return: ent_indices: shape num_mentions x num_neg_per_mention  array containing entity indices
			entities tensor containing tokenized entities : shape num_mentions x num_neg_per_mention x entity_len
			and entity scores tensor with shape num_mentions x num_neg_per_mention
	"""
	
	with open(ent_w_scores_file, "r") as fin:
		data = json.load(fin)
	
	ent_indices, ent_scores = np.array(data["indices"]), np.array(data["scores"])
	
	ent_indices, ent_scores = _sort_by_score(indices=ent_indices, scores=ent_scores)
	
	ent_indices = ent_indices[:n_ments, :num_labels]
	ent_scores = ent_scores[:n_ments, :num_labels]
	
	assert ent_indices.shape == (n_ments, num_labels), f"ent_indices shape = {ent_indices.shape} " \
													   f"does not match n_ments, num_labels = {n_ments, num_labels}"
	assert ent_indices.shape == ent_scores.shape, f"Indices arrays shape = {ent_indices.shape} is different " \
												  f"from score array shape = {ent_scores.shape}"
	tkn_labels_for_distill = []
	for ment_idx in range(n_ments):
		# Accumulate tokenizations of neg labels/entities for this mention
		curr_entities = [tokenized_entities[curr_ent_idx].unsqueeze(0)
						 for curr_ent_idx in ent_indices[ment_idx]]
		tkn_labels_for_distill += [torch.cat(curr_entities).unsqueeze(0)]

	tkn_labels_for_distill = torch.cat(tkn_labels_for_distill) # Shape : num_mentions x num_neg_per_mention x entity_len
	
	return ent_indices, tkn_labels_for_distill, ent_scores


def get_dataloader(config, raw_data, batch_size, shuffle_data, biencoder):
	"""
	Create pytorch dataloader object with given raw_data
	:param config:
	:param raw_data: Dict mapping domain identifies to (mention_data, entity_data) tuple
	:param batch_size:
	:param biencoder: Pass None if not using hard negatives
	:param shuffle_data: Shuffle data in dataloaders
	
	:return: Object of type DataLoader
	"""
	tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.lowercase)
	
	if config.data_type == "ent_link":
		all_datasets = []
		for domain in sorted(raw_data):
			
			if domain in config.trn_files:
				entity_file = config.trn_files[domain][1]
				ent_tokens_file = config.trn_files[domain][2]
			elif domain in config.dev_files:
				entity_file = config.dev_files[domain][1]
				ent_tokens_file = config.dev_files[domain][2]
			else:
				raise NotImplementedError(f"Domain ={domain} not present in "
										  f"train domains = {list(config.trn_files.keys())} or "
										  f"dev domains = {list(config.trn_files.keys())}")
			
			dataset = get_ent_link_dataset(
				raw_data=raw_data[domain],
				tokenizer=tokenizer,
				neg_strategy=config.neg_strategy,
				ent_tokens_file=ent_tokens_file,
				model_type=config.model_type,
				max_input_len=config.max_input_len,
				max_label_len=config.max_label_len,
				num_negs=config.num_negs,
				ent_w_score_file=config.ent_w_score_file_template.format(domain),
				biencoder=biencoder,
				entity_file=entity_file,
			)
			
			all_datasets += [dataset]
		
		if isinstance(all_datasets[0], TensorDataset) or isinstance(all_datasets[0], ConcatDataset):
			all_datasets = ConcatDataset(all_datasets)
			dataloader = DataLoader(
				dataset=all_datasets,
				shuffle=shuffle_data,
				batch_size=batch_size
			)
			return dataloader
		else:
			raise Exception(f"Invalid type of dataset = {type(all_datasets[0])}")
	
	elif config.data_type == "ent_link_ce":
		all_datasets = []
		
		for split_n_domain in sorted(raw_data):
			# (mention_data, entity_data) = raw_data[domain]
			# (title2id, id2title, id2text, kb_id2local_id) = entity_data
			split, domain = split_n_domain.split("~")
			
			ent_tokens_file = config.entity_token_file_template.format(domain)
			if split == "train":
				ent_w_score_file = config.train_ent_w_score_file_template.format(domain)
			elif split == "dev":
				ent_w_score_file = config.dev_ent_w_score_file_template.format(domain)
			else:
				raise NotImplementedError(f"split={split} not supported")
			
			dataset = get_ent_link_ce_dataset(
				raw_data=raw_data[split_n_domain],
				tokenizer=tokenizer,
				neg_strategy=config.neg_strategy,
				ent_tokens_file=ent_tokens_file,
				model_type=config.model_type,
				max_input_len=config.max_input_len,
				max_label_len=config.max_label_len,
				seed=config.seed,
				ent_w_score_file=ent_w_score_file,
				biencoder=biencoder,
				num_pos_labels_for_distill=config.distill_n_labels,
			)
			
			all_datasets += [dataset]
		
		if isinstance(all_datasets[0], TensorDataset) or isinstance(all_datasets[0], ConcatDataset):
			all_datasets = ConcatDataset(all_datasets)
			dataloader = DataLoader(
				dataset=all_datasets,
				shuffle=shuffle_data,
				batch_size=batch_size
			)
			return dataloader
		else:
			raise Exception(f"Invalid type of dataset = {type(all_datasets[0])}")
	
	else:
		raise Exception(f"Data type = {config.data_type} is not supported")


def get_ent_link_dataset(
		model_type,
		tokenizer,
		raw_data,
		ent_tokens_file,
		biencoder,
		neg_strategy,
		num_negs,
		max_input_len,
		max_label_len,
		ent_w_score_file,
		entity_file,
):
	"""
	Get dataset with tokenized data for entity linking using raw data.
	It first tokenizes the dataset and then creates a dataset with positive/negative training datapoints
	:param raw_data:
	:param tokenizer
	:param ent_tokens_file:
	:param model_type:
	:param max_input_len
	:param max_label_len:
	:param neg_strategy:
	:param num_negs:
	:param biencoder:
	:param entity_file:
	:param ent_w_score_file
	:return: Object of type TensorDataset
	"""
	try:
		mention_data, (title2id, id2title, id2text, kb_id2local_id) = raw_data
		
		#################################### TOKENIZE MENTIONS AND ENTITIES ############################################
		
		LOGGER.info("Loading and tokenizing mentions")
		tokenized_mentions = [get_context_representation(sample=mention,
														 tokenizer=tokenizer,
														 max_seq_length=max_input_len, )["ids"]
							  for mention in tqdm(mention_data)]
		LOGGER.info("Finished tokenizing mentions")
		tokenized_mentions = torch.LongTensor(tokenized_mentions)
		
		if ent_tokens_file is not None and os.path.isfile(ent_tokens_file):
			LOGGER.info(f"Reading tokenized entities from file {ent_tokens_file}")
			tokenized_entities = torch.LongTensor(np.load(ent_tokens_file))
		else:
			LOGGER.info(f"Tokenizing {len(id2title)} entities")
			tokenized_entities = [get_candidate_representation(candidate_title=id2title[ent_id],
															   candidate_desc=id2text[ent_id],
															   tokenizer=tokenizer,
															   max_seq_length=max_label_len)["ids"]
								  for ent_id in tqdm(sorted(id2title))]
			tokenized_entities = torch.LongTensor(tokenized_entities)
		
		################################################################################################################
		
		######################### GENERATE POSITIVE AND NEGATIVE LABELS FOR EACH DATAPOINT #############################
		
		# Creating list of list type pos_label_idxs because get_hard_negs function expects this format
		pos_label_idxs = [[int(mention["label_id"])] for mention in mention_data]
		n_labels = len(tokenized_entities)
		if neg_strategy == "random":
			neg_labels_idxs = get_random_negs(
				data=mention_data,
				seed=0,
				num_negs=num_negs,
				n_labels=n_labels,
				label_key="label_id"
			)
		elif neg_strategy == "bienc_hard_negs" and biencoder is None:
			warnings.warn("Mining negative randomly as biencoder model is not provided")
			neg_labels_idxs = get_random_negs(
				data=mention_data,
				seed=0,
				num_negs=num_negs,
				n_labels=n_labels,
				label_key="label_id"
			)
		elif neg_strategy == "bienc_hard_negs" and biencoder is not None:
			neg_labels_idxs, _ = get_hard_negs_biencoder(
				biencoder=biencoder,
				input_tokens_list=tokenized_mentions,
				labels_tokens_list=tokenized_entities,
				pos_label_idxs=pos_label_idxs,
				num_negs=num_negs
			)
		elif neg_strategy == "tfidf_hard_negs":
			neg_labels_idxs = get_hard_negs_tfidf(
				mentions_data=mention_data,
				entity_file=entity_file,
				pos_label_idxs=pos_label_idxs,
				num_negs=num_negs
			)
		elif neg_strategy == "in_batch":
			neg_labels_idxs = []
		elif neg_strategy == "precomp":
			# Load num_negs+1 labels form this file so that if gt label is present in this list,
			# we can remove it and still have num_negs negatives
			ent_indices, _, _ = get_precomputed_ents_w_scores(
				ent_w_scores_file=ent_w_score_file,
				num_labels=num_negs + 1,
				n_ments=len(tokenized_mentions),
				tokenized_entities=tokenized_entities
			)
			
			neg_labels_idxs = []
			# Remove pos-label from list of labels for each input/mention if it is present and finally keep num_negs labels
			for ment_idx, curr_pos_labels in enumerate(pos_label_idxs):
				temp_neg_labels_idxs = [curr_label_idx for curr_label_idx in ent_indices[ment_idx]
										if curr_label_idx not in curr_pos_labels][:num_negs]
				
				assert len(temp_neg_labels_idxs) > 0
				while len(temp_neg_labels_idxs) < num_negs:
					temp_neg_labels_idxs += temp_neg_labels_idxs
				# temp_neg_labels_idxs += [temp_neg_labels_idxs[-1]]*(num_negs - len(temp_neg_labels_idxs))
				
				neg_labels_idxs += [temp_neg_labels_idxs[:num_negs]]
		
		else:
			raise NotImplementedError(f"Negative sampling strategy = {neg_strategy} not implemented")
		
		# Simplifying list of single-element-list to list format
		pos_label_idxs = [curr_pos_labels[0] for curr_pos_labels in pos_label_idxs]
		dataset = _get_dataset_from_tokenized_inputs(
			model_type=model_type,
			tokenized_inputs=tokenized_mentions,
			tokenized_labels=tokenized_entities,
			pos_label_idxs=pos_label_idxs,
			neg_labels_idxs=neg_labels_idxs
		)
		
		return dataset
	except Exception as e:
		LOGGER.info(f"Exception raised in data_process.get_ent_link_dataset() {str(e)}")
		embed()
		raise e


def get_ent_link_ce_dataset(
		model_type,
		tokenizer,
		raw_data,
		ent_tokens_file,
		biencoder,
		neg_strategy,
		max_input_len,
		max_label_len,
		ent_w_score_file,
		num_pos_labels_for_distill,
		seed,
):
	"""
	Get dataset with tokenized data for training a model on precomputed mention-entity scores.
	:param raw_data:
	:param tokenizer
	:param ent_tokens_file:
	:param model_type:
	:param max_input_len
	:param max_label_len:
	:param neg_strategy:
	:param biencoder:
	:param num_pos_labels_for_distill:
	:param seed:
	:param ent_w_score_file
	:return: Object of type TensorDataset
	"""
	try:
		assert model_type == "bi_enc", f"Model_type = {model_type} is not supported in get_ent_link_ce_dataset function"
		mention_data, (title2id, id2title, id2text, kb_id2local_id) = raw_data
		
		#################################### TOKENIZE MENTIONS AND ENTITIES ############################################
		LOGGER.info("Loading and tokenizing mentions")
		all_tokenized_mentions = [get_context_representation(sample=mention,
															 tokenizer=tokenizer,
															 max_seq_length=max_input_len, )["ids"]
								  for mention in tqdm(mention_data)]
		LOGGER.info("Finished tokenizing mentions")
		all_tokenized_mentions = torch.LongTensor(all_tokenized_mentions)
		
		if ent_tokens_file is not None and os.path.isfile(ent_tokens_file):
			LOGGER.info(f"Reading tokenized entities from file {ent_tokens_file}")
			tokenized_entities = torch.LongTensor(np.load(ent_tokens_file))
		else:
			LOGGER.info(f"Tokenizing {len(id2title)} entities")
			tokenized_entities = [get_candidate_representation(candidate_title=id2title[ent_id],
															   candidate_desc=id2text[ent_id],
															   tokenizer=tokenizer,
															   max_seq_length=max_label_len)["ids"]
								  for ent_id in tqdm(sorted(id2title))]
			tokenized_entities = torch.LongTensor(tokenized_entities)
		
		################################## READ CROSSENCODER SCORES FROM FILE  #########################################
		LOGGER.info(f"Read scores from  file ={ent_w_score_file}")
		with open(ent_w_score_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			
			ment_to_ent_scores = dump_dict["ment_to_ent_scores"]
			mention_tokens_list = dump_dict["mention_tokens_list"]
			entity_id_list = dump_dict["entity_id_list"]
			ment_idxs = dump_dict["ment_idxs"] if "ment_idxs" in dump_dict else np.arange(ment_to_ent_scores.shape[0])
			
			# mention_data = dump_dict["test_data"]
			# entity_tokens_list = dump_dict["entity_tokens_list"]
			# arg_dict = dump_dict["arg_dict"]
		
		n_ments, n_ents = ment_to_ent_scores.shape
		tokenized_mentions = torch.LongTensor(mention_tokens_list)
		assert (len(entity_id_list) == 0) or (entity_id_list == np.arange(n_ents)).all(), f"len(entity_id_list) = {len(entity_id_list)} and it needs to be coupled with entity scores"
		assert (tokenized_mentions == all_tokenized_mentions[ment_idxs]).all(), f"Error in mention tokenization, tokenized_mentions.shape={tokenized_mentions.shape}, "
		
		LOGGER.info(f"Sorting scores for each mention")
		ent_indices = np.vstack([np.arange(n_ents) for _ in range(n_ments)])  # shape = n_ments, n_ents
		assert ent_indices.shape == ment_to_ent_scores.shape, f"ent_indices.shape = {ent_indices.shape} != ment_to_ent_scores.shape = {ment_to_ent_scores.shape}"

		sorted_ent_indices, sorted_ent_scores = _sort_by_score(indices=ent_indices, scores=ment_to_ent_scores)

		top_ent_indices = sorted_ent_indices[:, :num_pos_labels_for_distill]
		top_ent_scores = sorted_ent_scores[:, :num_pos_labels_for_distill]

		assert top_ent_indices.shape == (n_ments, num_pos_labels_for_distill), f"ent_indices shape = {top_ent_indices.shape} does not match n_ments, num_labels = {n_ments, num_pos_labels_for_distill}"
		assert top_ent_indices.shape == top_ent_scores.shape, f"Indices arrays shape = {top_ent_indices.shape} is different from score array shape = {top_ent_scores.shape}"

		################################################################################################################
		
		######################### GENERATE POSITIVE AND NEGATIVE LABELS FOR EACH DATAPOINT #############################
		
		LOGGER.info(f"Loading negs for strategy = {neg_strategy}")
		if neg_strategy in ["top_ce_match"]:
			tkn_labels_for_distill = []
			for curr_top_ent_indices in top_ent_indices:
				# Accumulate tokenizations of top labels/entities for this mention
				curr_entities = [tokenized_entities[curr_ent_idx].unsqueeze(0) for curr_ent_idx in curr_top_ent_indices]
				tkn_labels_for_distill += [torch.cat(curr_entities).unsqueeze(0)]
		
			tkn_labels_for_distill = torch.cat(tkn_labels_for_distill) # Shape : num_mentions x num_top_ents_per_mention x entity_seq_len
			
			LOGGER.info(f"top_ent_scores.shape = {top_ent_scores.shape}")
			LOGGER.info(f"tokenized_mentions.shape = {tokenized_mentions.shape}")
			LOGGER.info(f"tkn_labels_for_distill.shape = {tkn_labels_for_distill.shape}")
			dataset = TensorDataset(tokenized_mentions, tkn_labels_for_distill, top_ent_scores)
			return dataset
		
		elif neg_strategy in ["top_ce_w_bienc_hard_negs_trp", "top_ce_w_rand_negs_trp"]:
			
			# Triplet style - Find num_pos_labels_for_distill negatives and then pair each with one positive label (i.e. label with topk-ce scores)
			
			if biencoder is None or neg_strategy == "top_ce_w_rand_negs_trp":
				warnings.warn(f"Mining negative randomly as biencoder model is not provided or neg_strategy = {neg_strategy}")
				neg_labels_idxs = get_random_negs_w_blacklist(
					n_data=len(ment_idxs),
					seed=seed,
					num_negs=num_pos_labels_for_distill,
					n_labels=len(tokenized_entities),
					label_blacklist=top_ent_indices
				)
			else:
				# Get hard negatives for biencoder while treating top-cross-encoder labels as positive labels
				neg_labels_idxs, _ = get_hard_negs_biencoder(
					biencoder=biencoder,
					input_tokens_list=tokenized_mentions,
					labels_tokens_list=tokenized_entities,
					pos_label_idxs=top_ent_indices,
					num_negs=num_pos_labels_for_distill
				)
			
			trp_ment_tokens = []
			trp_pos_tokens = []
			trp_neg_tokens = []
			for ment_iter in range(n_ments):
				for label_iter in range(num_pos_labels_for_distill):
					curr_pos_ent_idx = top_ent_indices[ment_iter][label_iter]
					curr_neg_ent_idx = neg_labels_idxs[ment_iter][label_iter]
					
					trp_ment_tokens += [tokenized_mentions[ment_iter]]
					trp_pos_tokens += [tokenized_entities[curr_pos_ent_idx]]
					trp_neg_tokens += [tokenized_entities[curr_neg_ent_idx]]
			
			trp_ment_tokens = torch.stack(trp_ment_tokens)  # shape: num_mentions*num_pos_labels_for_distill, seq_len
			trp_pos_tokens = torch.stack(trp_pos_tokens)  # shape: num_mentions*num_pos_labels_for_distill, seq_len
			trp_neg_tokens = torch.stack(trp_neg_tokens)  # shape: num_mentions*num_pos_labels_for_distill, seq_len
			
			# shape: num_mentions*num_pos_labels_for_distill, 1,  seq_len -
			# This allows us to use same biencoder forward function as used when using larger number of negatives per mention and training biencoder with ground-truth entity data
			trp_neg_tokens = trp_neg_tokens.unsqueeze(1)
			
			LOGGER.info(f"trp_ment_tokens.shape = {trp_ment_tokens.shape}")
			LOGGER.info(f"trp_pos_tokens.shape = {trp_pos_tokens.shape}")
			LOGGER.info(f"trp_neg_tokens.shape = {trp_neg_tokens.shape}")
		
			dataset = TensorDataset(trp_ment_tokens, trp_pos_tokens, trp_neg_tokens)
			
			return dataset
		

		else:
			raise NotImplementedError(f"neg_strategy = {neg_strategy} not supported in get_ent_link_ce_dataset()")
			
	except Exception as e:
		LOGGER.info(f"Exception raised in data_process.get_ent_link_dataset() {str(e)}")
		embed()
		raise e


def _get_dataset_from_tokenized_inputs(model_type, tokenized_inputs, tokenized_labels, pos_label_idxs, neg_labels_idxs):
	"""
	Helper function that creates dataset containing positive/negative examples
	using already tokenized labels and inputs.
	
	:param model_type:
	:param tokenized_inputs:
	:param tokenized_labels:
	:param pos_label_idxs: List of positive labels for corresponding input
	:param neg_labels_idxs: List of which contains list of negative labels for each input
	:return: Object of type TensorDataset
	"""
	
	try:
		LOGGER.info(f"Shape of tokenized_labels = {tokenized_labels.shape}")
		
		if model_type == "bi_enc":
			tokenized_pos_label = tokenized_labels[pos_label_idxs]
			if len(neg_labels_idxs) == 0:
				tokenized_tensor_data = TensorDataset(tokenized_inputs, tokenized_pos_label)
			else:
				tokenized_neg_labels = []
				for idx in range(len(tokenized_inputs)):
					# Accumulate tokenizations of neg labels/entities for this mention
					curr_neg_labels = [tokenized_labels[neg_idx].unsqueeze(0) for neg_idx in neg_labels_idxs[idx]]
					tokenized_neg_labels += [torch.cat(curr_neg_labels).unsqueeze(0)]
	
				tokenized_neg_labels = torch.cat(tokenized_neg_labels) # Shape : num_mentions x num_neg_per_mention x entity_len
				tokenized_tensor_data = TensorDataset(tokenized_inputs, tokenized_pos_label, tokenized_neg_labels)
			
		elif model_type == "cross_enc":
			
			pos_paired_token_idxs, neg_paired_token_idxs = _get_paired_token_idxs(tokenized_inputs, tokenized_labels,
																				  pos_label_idxs, neg_labels_idxs)
			pos_paired_token_idxs = torch.cat(pos_paired_token_idxs)
			neg_paired_token_idxs = torch.cat(neg_paired_token_idxs)
			tokenized_tensor_data = TensorDataset(pos_paired_token_idxs, neg_paired_token_idxs)
		else:
			raise NotImplementedError(f"Data loading for model_type = {model_type} not supported.")
		
		return tokenized_tensor_data
	except Exception as e:
		embed()
		raise e


def _get_paired_token_idxs(tokenized_inputs, tokenized_labels, pos_label_idxs, neg_labels_idxs):
	"""
	Concatenates input and label tokens using pos_label_idxs to create a tensor of pos_paired_token_idxs, and
	Concatenates input and label tokens using neg_label_idxs to create a tensor of neg_paired_token_idxs
	:param tokenized_inputs:
	:param tokenized_labels:
	:param pos_label_idxs: List of positive label per input
	:param neg_labels_idxs: List of list of negative labels per input
	:return: List of input-paired-with-pos-labels, and list of input-paired-with-neg-labels
	"""
	
	tokenized_pos_label = tokenized_labels[pos_label_idxs]
	pos_paired_token_idxs = []
	neg_paired_token_idxs = []
	for idx, input_tkns in enumerate(tokenized_inputs):
		pos_label_tkns = tokenized_pos_label[idx]
		# Create paired rep for current mentions with positive/ground-truth entities/labels
		pos_pair_rep = create_input_label_pair(input_token_idxs=input_tkns, label_token_idxs=pos_label_tkns)
		pos_paired_token_idxs += [pos_pair_rep.unsqueeze(0)]
		
		# Create paired rep for current mentions with all negative entities/labels
		curr_neg_pair_reps = []
		for neg_idx in neg_labels_idxs[idx]:
			curr_neg_pair = create_input_label_pair(input_token_idxs=input_tkns, label_token_idxs=tokenized_labels[neg_idx])
			curr_neg_pair_reps += [curr_neg_pair.unsqueeze(0)]
		
		curr_neg_pair_reps = torch.cat(curr_neg_pair_reps)
		neg_paired_token_idxs += [curr_neg_pair_reps.unsqueeze(0)]
	
	return pos_paired_token_idxs, neg_paired_token_idxs
	

def create_input_label_pair(input_token_idxs, label_token_idxs):
	"""
	Remove cls token from label (this is the first token) and concatenate with input
	:param input_token_idxs:
	:param label_token_idxs:
	:return:
	"""
	if isinstance(input_token_idxs, torch.Tensor):
		return torch.cat((input_token_idxs, label_token_idxs[1:]))
	else: # numpy arrays support concat with + operator
		return input_token_idxs + label_token_idxs[1:]


######## Function from blink/biencoder.data_process.py ######


def get_context_representation(
		sample,
		tokenizer,
		max_seq_length,
		mention_key="mention",
		context_key="context",
		ent_start_token=ENT_START_TAG,
		ent_end_token=ENT_END_TAG,
):
	mention_tokens = []
	if sample[mention_key] and len(sample[mention_key]) > 0:
		mention_tokens = tokenizer.tokenize(sample[mention_key])
		mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
	
	context_left = sample[context_key + "_left"]
	context_right = sample[context_key + "_right"]
	context_left = tokenizer.tokenize(context_left)
	context_right = tokenizer.tokenize(context_right)
	
	left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
	right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
	left_add = len(context_left)
	right_add = len(context_right)
	if left_add <= left_quota:
		if right_add > right_quota:
			right_quota += left_quota - left_add
	else:
		if right_add <= right_quota:
			left_quota += right_quota - right_add
	
	context_tokens = (
			context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
	)
	
	context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
	input_ids = tokenizer.convert_tokens_to_ids(context_tokens)[:max_seq_length]
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	assert len(input_ids) == max_seq_length, f"Input_ids len = {len(input_ids)} != max_seq_len ({max_seq_length})"
	
	return {
		"tokens": context_tokens,
		"ids": input_ids,
	}


def get_candidate_representation(
		candidate_desc,
		tokenizer,
		max_seq_length,
		candidate_title=None,
		title_tag=ENT_TITLE_TAG,
):
	try:
		cls_token = tokenizer.cls_token
		sep_token = tokenizer.sep_token
		cand_tokens = tokenizer.tokenize(candidate_desc)
		if candidate_title is not None:
			title_tokens = tokenizer.tokenize(candidate_title)
			cand_tokens = title_tokens + [title_tag] + cand_tokens
		
		cand_tokens = cand_tokens[: max_seq_length - 2]
		cand_tokens = [cls_token] + cand_tokens + [sep_token]
		
		input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		assert len(input_ids) == max_seq_length
		
		return {
			"tokens": cand_tokens,
			"ids": input_ids,
		}
	except Exception as e:
		embed()
		raise e
