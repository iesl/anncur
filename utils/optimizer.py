import sys
import logging

from torch.optim import SGD
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.optimization import WarmupLinearSchedule

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)

patterns_optimizer = {
	'additional_layers': ['additional'],
	'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
	'top4_layers': [
		'additional',
		'bert_model.encoder.layer.11.',
		'encoder.layer.10.',
		'encoder.layer.9.',
		'encoder.layer.8',
	],
	'all_encoder_layers': ['additional', 'bert_model.encoder.layer'],
	'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings'],
	'embeddings': ['bert_model.embeddings']
}


def get_bert_optimizer(models, type_optimization, learning_rate, weight_decay, optimizer_type, verbose=True):
	""" Optimizes the network with AdamWithDecay or SGD depending on optimizer_type
	"""
	if type_optimization not in patterns_optimizer:
		LOGGER.info(f'Error. Type optimizer must be one of {str(patterns_optimizer.keys())}')
	parameters_with_decay = []
	parameters_with_decay_names = []
	parameters_without_decay = []
	parameters_without_decay_names = []
	no_decay = ['bias', 'gamma', 'beta']
	patterns = patterns_optimizer[type_optimization]

	for model in models:
		for n, p in model.named_parameters():
			if any(t in n for t in patterns):
				if any(t in n for t in no_decay):
					parameters_without_decay.append(p)
					parameters_without_decay_names.append(n)
				else:
					parameters_with_decay.append(p)
					parameters_with_decay_names.append(n)
	
	if verbose:
		LOGGER.info('The following parameters will be optimized WITH decay:')
		LOGGER.info(ellipse(parameters_with_decay_names, 5, ' , '))
		LOGGER.info('The following parameters will be optimized WITHOUT decay:')
		LOGGER.info(ellipse(parameters_without_decay_names, 5, ' , '))

	optimizer_grouped_parameters = [
		{'params': parameters_with_decay, 'weight_decay': weight_decay},
		{'params': parameters_without_decay, 'weight_decay': 0.0},
	]
	
	if optimizer_type == "SGD":
		optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)
	elif optimizer_type == "AdamW":
		optimizer = AdamW(
			optimizer_grouped_parameters,
			lr=learning_rate,
			correct_bias=False
		)
	else:
		raise NotImplementedError(f"Optimizer_type = {optimizer_type} not supported")
	
	return optimizer


def get_scheduler(optimizer, epochs, warmup_proportion, len_data, batch_size, grad_acc_steps):
	
	num_train_steps = int(len_data / int(batch_size / grad_acc_steps)) * epochs
	num_warmup_steps = int(num_train_steps * warmup_proportion)

	scheduler = WarmupLinearSchedule(
		optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
	)
	LOGGER.info(f" Num optimization steps = {num_train_steps}")
	LOGGER.info(f" Num warmup steps = {num_warmup_steps}")
	return scheduler


def ellipse(lst, max_display=5, sep='|'):
	"""
	Like join, but possibly inserts an ellipsis.
	:param lst: The list to join on
	:param int max_display: the number of items to display for ellipsing.
		If -1, shows all items
	:param string sep: the delimiter to join on
	"""
	# copy the list (or force it to a list if it's a set)
	choices = list(lst)
	# insert the ellipsis if necessary
	if max_display > 0 and len(choices) > max_display:
		ellipsis = '...and {} more'.format(len(choices) - max_display)
		choices = choices[:max_display] + [ellipsis]
	return sep.join(str(c) for c in choices)
