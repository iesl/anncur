import os
import sys
from pathlib import Path




def save_code(config, out_dir = "code"):
	code_dir = "{}/{}".format(config.result_dir, out_dir)
	command = "rsync -avi --exclude=__pycache__ --exclude=slurm*.out --exclude=*.ipynb " \
			  "--exclude=.ipynb_checkpoints --exclude=.gitignore ../cross-encoder-xmc/  {}/".format(code_dir)
	
	Path(code_dir).mkdir(parents=True, exist_ok=True)  # Create result_dir directory if not already present
	os.system(command)
	command = "echo {}  > {}/command.txt".format(" ".join(sys.argv), code_dir)
	os.system(command)
