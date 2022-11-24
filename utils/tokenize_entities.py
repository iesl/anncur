import os
import sys
import argparse

from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path
from pytorch_transformers.tokenization_bert import BertTokenizer

from utils.data_process import load_entities, get_candidate_representation
logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def main(ent_file, out_file, bert_model_type, max_seq_len, use_lowercase):
	
	LOGGER.info(f"Tokenizing entities from file {ent_file}")
	(title2id,
	 id2title,
	 id2text,
	 kb_id2local_id) = load_entities(entity_file=ent_file)
	
	tokenizer = BertTokenizer.from_pretrained(bert_model_type, do_lower_case=use_lowercase)
	
	tokenized_entities = [ get_candidate_representation(candidate_title=id2title[ent_id],
														candidate_desc=id2text[ent_id],
														tokenizer=tokenizer,
														max_seq_length=max_seq_len)["ids"]
							for ent_id in tqdm(sorted(id2title))]
	tokenized_entities = np.array(tokenized_entities)
	out_dir = os.path.dirname(out_file)
	Path(out_dir).mkdir(exist_ok=True, parents=True)
	
	np.save(file=out_file, arr=tokenized_entities)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Split zeshel data into separate folder/files wrt worlds')
	
	parser.add_argument("--ent_file", type=str, required=True, help="File containing entities to tokenize")
	parser.add_argument("--out_file", type=str, required=True, help="Output file containing tokenized entities")
	
	parser.add_argument("--bert_model_type", type=str, required=True, help="Type of bert model eg bert-base-uncased")
	parser.add_argument("--max_seq_len", type=int, required=True, help="Maximum seq len for tokenized entities")
	parser.add_argument("--lowercase", type=int, required=True, help="Lowercase tokenizer? 0 for false, 1 for true")
	
	
	args = parser.parse_args()
	
	_ent_file = args.ent_file
	_out_file = args.out_file
	_max_seq_len = args.max_seq_len
	_bert_model_type = args.bert_model_type
	_lowercase = bool(args.lowercase)
	
	main(ent_file=_ent_file, out_file=_out_file,
		 bert_model_type=_bert_model_type, max_seq_len=_max_seq_len, use_lowercase=_lowercase)
