import os
import sys
import json
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def preprocess_zeshel_data(root_data_dir):
	"""
	Create train/dev/test files from raw zeshel data from https://github.com/lajanugen/zeshel
	:param root_data_dir: dir containing 'documents' and 'mentions' folders. This is usually dir that is created after extracting raw ZeShEL data
	:return:
	"""
	# This file converts zeshel style data into the format BLINK expects
	#
	# Keys for each mention:
	#   - mention
	#   - context_left
	#   - context_right
	#   - label_id
	#   - world (zeshel only)
	#   - label
	#   - label_title
	
	

	DATA_DIR = root_data_dir
	OUTPUT_DIR = os.path.join(root_data_dir, 'processed')
	
	Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
	# get all of the documents
	entity2idx = {}
	documents = {}
	doc_dir = os.path.join(DATA_DIR, 'documents')
	for doc_fname in tqdm(os.listdir(doc_dir), desc='Loading documents'):
		assert doc_fname.endswith('.json')
		with open(os.path.join(doc_dir, doc_fname), 'r') as f:
			for idx, line in enumerate(f):
				one_doc = json.loads(line.strip())
				doc_id = one_doc['document_id']
				assert doc_id not in documents
				documents[doc_id] = one_doc
	
	
	# get all of the train mentions
	print('Processing mentions...')
	
	splits = ['train', 'val', 'test']
	
	for split in splits:
		blink_mentions = []
		with open(os.path.join(DATA_DIR, 'mentions', split + '.json'), 'r') as f:
			for line in tqdm(f):
				one_mention = json.loads(line.strip())
				label_doc = documents[one_mention['label_document_id']]
				context_doc = documents[one_mention['context_document_id']]
				start_index = one_mention['start_index']
				end_index = one_mention['end_index']
				context_tokens = context_doc['text'].split()
				extracted_mention = ' '.join(context_tokens[start_index:end_index+1])
				assert extracted_mention == one_mention['text']
				context_left = ' '.join(context_tokens[:start_index])
				context_right = ' '.join(context_tokens[end_index+1:])
				transformed_mention = {}
				transformed_mention['mention'] = extracted_mention
				transformed_mention['mention_id'] = one_mention['mention_id']
				transformed_mention['context_left'] = context_left
				transformed_mention['context_right'] = context_right
				transformed_mention['context_doc_id'] = one_mention['context_document_id']
				transformed_mention['type'] = one_mention['corpus']
				transformed_mention['label_id'] = one_mention['label_document_id']
				transformed_mention['label'] = label_doc['text']
				transformed_mention['label_title'] = label_doc['title']
				blink_mentions.append(transformed_mention)
		print('Done.')
		# write all of the transformed train mentions
		print('Writing processed mentions to file...')
		with open(os.path.join(OUTPUT_DIR, split + '.jsonl'), 'w') as f:
			f.write('\n'.join([json.dumps(m) for m in blink_mentions]))
		print('Done.')

	
def split_files(data_fname, out_dir):
	"""
	Split data in each split into separate domain/worlds in ZeShEL
	:param data_fname:
	:param out_dir:
	:return:
	"""

	world_to_ments = defaultdict(list)
	with open(data_fname, "r") as reader:
		for line in reader:
			ment_dict  = json.loads(line.strip())

			world_to_ments[ment_dict["type"]] += [ment_dict]


	LOGGER.info("Writing mentions for each world separately")
	Path(out_dir).mkdir(exist_ok=True, parents=True)
	for world in world_to_ments:
		with open(f"{out_dir}/{world}_mentions.jsonl", "w") as writer:
			for ment in world_to_ments[world]:
				writer.write(json.dumps(ment) + "\n")



if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Preprocess ZeShEL data')

	parser.add_argument("--root_data_dir", type=str, required=True, help="Dir containing zeshel data. This should contain 'documents' and 'mentions' folder.")
	args = parser.parse_args()

	root_data_dir = args.root_data_dir
	
	preprocess_zeshel_data(root_data_dir=root_data_dir)
	
	# This creates three files under <root_data_dir>/processed -- train.jsonl, test.jsonl, valid.jsonl
	splits = ["train", "test", "val"]
	for split in splits:
		data_fname = f"{root_data_dir}/processed/{split}.jsonl"
		out_dir = f"{root_data_dir}/processed/{split}_worlds"
		print(f"Processing split={split}, \nfile = {data_fname} and \nstoring res in {out_dir}")
		split_files(
			data_fname=data_fname,
			out_dir=out_dir,
		)
	
	
	# Rename val_worlds to valid_worlds
	out_dir_1 = f"{root_data_dir}/processed/val_worlds"
	out_dir_2 = f"{root_data_dir}/processed/valid_worlds"
	
	os.system(f"mv {out_dir_1} {out_dir_2}")
	
	# Rename val_worlds to valid_worlds
	file1 = f"{root_data_dir}/processed/val.jsonl"
	file2 = f"{root_data_dir}/processed/valid.jsonl"
	
	os.system(f"mv {file1} {file2}")
