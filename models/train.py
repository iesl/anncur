import sys
import time
import logging
import argparse

from pathlib import Path

from models.pairwise_trainer import BasePairwiseTrainer
from utils.config import Config
from utils.basic_utils import save_code

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def get_trainer(config):
	return BasePairwiseTrainer(config=config)


def run(config):
	
	assert isinstance(config, Config)
	
	command = sys.argv
	start = time.time()
	
	trainer = get_trainer(config)
	LOGGER.addHandler(logging.FileHandler(f"{trainer.config.result_dir}/log_file.txt"))
	
	LOGGER.info("COMMAND = {}".format(command))
	
	if config.mode == "train":
		train_t1 = time.time()
		trainer.train()
		train_t2 = time.time()
		LOGGER.info("Training ends in time={:.3f} = {:.3f} min = {:.3f} hr Saving model".format(train_t2 - train_t1,(train_t2 - train_t1)/60,(train_t2 - train_t1)/3600))
	else:
		raise Exception("Invalid mode = {}".format(config.mode))
	
	LOGGER.info(trainer)
	LOGGER.info(command)
	end = time.time()
	LOGGER.info(" Total time taken  = {:.4f} = {:.4f} min = {:.4f} hours".format(end - start, (end - start)/60, (end - start)/3600))


def main():
	parser = argparse.ArgumentParser( description='Script to train a pairwise scoring function')
	parser.add_argument("--config", type=str, required=True, help="config file")
	args, remaining_args = parser.parse_known_args()
	
	config = Config(args.config)
	config.update_config_from_arg_list(arg_list=remaining_args)
	Path(config.result_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present
	config.save_config(config.result_dir, "orig_config.json")
	
	if config.save_code: save_code(config)
	
	run(config)


if __name__ == '__main__':
	main()

