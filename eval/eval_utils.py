import sys
import torch
import logging
import numpy as np

from tqdm import tqdm
from IPython import embed
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_reci_rank(gt, preds, scores):
	"Get reciprocal rank of gt in preds array sorted using scores"
	
	sorted_preds_scores = sorted(list(zip(preds, scores)), key=lambda x:x[1], reverse=True)

	preds, scores = zip(*sorted_preds_scores)
	for pred_idx, curr_pred in enumerate(preds):
		if curr_pred == gt:
			return 1./(pred_idx + 1.)
		
	return 0.



def score_topk_preds(gt_labels, topk_preds):
	"""
	Evaluate scores for top_k preds
	:param gt_labels:
	:param topk_preds: Dict with keys indices, and scores containig 2-D arrays storing predicted indices and scores for each input
	:return:
	"""
	res = []
	for idx,curr_gt in enumerate(gt_labels):
		res += [get_reci_rank(gt=curr_gt,
							  preds=topk_preds["indices"][idx],
							  scores=topk_preds["scores"][idx])]
	
	res = np.array(res)
	res_metrics =  {"acc": "{:.2f}".format(100*np.mean(res == 1)),
					"mrr": "{:.2f}".format(100*np.mean(res)),
					"recall": "{:.2f}".format(100*np.mean(res > 0)),
					"recall_5": "{:.2f}".format(100*np.mean(res > 1/6)),
					"recall_10": "{:.2f}".format(100*np.mean(res > 1/11)),
					"recall_64": "{:.2f}".format(100*np.mean(res > 1/65)),
					"norm_acc": "{:.2f}".format(100*np.mean(res[res > 0] == 1)),
					"norm_mrr": "{:.2f}".format(100*np.mean(res[res > 0]))
					}
	return res_metrics


def compute_embeddings_w_biencoder(biencoder, data_tokens_list, batch_size, input_type):
	
	from models.biencoder import BiEncoderWrapper
	from models.crossencoder import CrossEncoderWrapper
	batched_data = TensorDataset(data_tokens_list)
	bienc_dataloader = DataLoader(batched_data, batch_size=batch_size, shuffle=False)
	
	if isinstance(biencoder, torch.nn.parallel.distributed.DistributedDataParallel):
		biencoder = biencoder.module.module
		
	assert isinstance(biencoder, BiEncoderWrapper) or isinstance(biencoder, CrossEncoderWrapper), f"Expected model of type = BiEncoderWrapper but got of type = {type(biencoder)}"
	with torch.no_grad():
		# biencoder.eval() -- Avoid calling .eval() here because if this is called during training, we need to also call model.train(). It is simpler to assume model is in eval mode already
		assert not biencoder.training, "Biencoder model should be in eval mode"
		
		all_encodings = []
		LOGGER.info(f"Starting embedding data with n_data={len(data_tokens_list)}")
		LOGGER.info(f"Bi encoder model device {biencoder.device}")
		for batch_idx, (batch_data,) in tqdm(enumerate(bienc_dataloader), position=0, leave=True, total=len(bienc_dataloader)):
			batch_data =  batch_data.to(biencoder.device)
			if input_type == "label":
				encodings = biencoder.encode_label(batch_data)
			elif input_type == "input":
				encodings = biencoder.encode_input(batch_data)
			else:
				raise Exception(f"Encoding data of data type = {input_type} not supported")
			
			all_encodings += [encodings.cpu()]
			torch.cuda.empty_cache()
			
		all_encodings = torch.cat(all_encodings)
		
		
	return all_encodings


def compute_input_embeddings(biencoder, input_tokens_list, batch_size):
	
	return compute_embeddings_w_biencoder(
		biencoder=biencoder,
		data_tokens_list=input_tokens_list,
		batch_size=batch_size,
		input_type="input"
	)


def compute_label_embeddings(biencoder, labels_tokens_list, batch_size):
	
	return compute_embeddings_w_biencoder(
		biencoder=biencoder,
		data_tokens_list=labels_tokens_list,
		batch_size=batch_size,
		input_type="label"
	)


def compute_overlap(indices_list1, indices_list2):
	"""
	Comute overlap metrics b/w corresponding pairs of lists
	:param indices_list1:
	:param indices_list2:
	:return:
	"""
	all_res = []
	for indices1, indices2 in zip(indices_list1, indices_list2):
		res = _compute_overlap_helper(indices1=indices1, indices2=indices2)
		all_res += [res]
	
	metrics = ["common", "diff", "total", "common_frac", "diff_frac"]
	avg_res = {}
	if len(all_res) == 0:
		return {metric: ("mean 0.0", "std 0.0", "p50 0.0") for metric in metrics}
	
	for metric in metrics:
		mean = np.mean([res[metric] for res in all_res])
		std = np.std([res[metric] for res in all_res])
		p50 = np.percentile([res[metric] for res in all_res], 50)
		avg_res[metric] = "mean {:.4f}".format(mean), "std {:.4f}".format(std), "p50 {:.4f}".format(p50)
		
	return avg_res
	

def _compute_overlap_helper(indices1, indices2):
	n_intersection = len(set(indices1).intersection(set(indices2)))
	n_total = len(indices1) + len(indices2)
	assert len(indices1) == len(indices2), f"Len of both indices is not same => {len(indices1)} != {len(indices2)}"
	
	n = len(indices1)
	n_diff = n - n_intersection
	
	return {"common":n_intersection, "diff":n_diff, "total":n,
			"common_frac":n_intersection/n, "diff_frac":n_diff/n}
	