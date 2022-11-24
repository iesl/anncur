import os
import sys
import copy
import json
import pickle
import logging
import argparse
import csv
import glob
import torch
import itertools

from IPython import embed
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm

from utils.zeshel_utils import get_zeshel_world_info, N_ENTS_ZESHEL as NUM_ENTS

plt.rcParams.update({
  "text.usetex": True,
  # "font.family": "Helvetica"
})

logging.basicConfig(
	stream=sys.stderr,
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
	datefmt="%d/%m/%Y %H:%M:%S",
	level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)
	
cmap = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
	('navy', 'skyblue'), ('olive', 'y'),
	('sienna', 'tan'), ('darkviolet', 'orchid'),
	('darkorange', 'gold'), ('deeppink', 'violet'),
	('deepskyblue', 'lightskyblue'), ('gray', 'silver')]



def plot_rq_1_2_performance_vs_topk_retrieved_or_cost(base_res_dir, nm_train, domain, rq_name, topk_vals, eval_data="test"):
	
	try:
		# font = {'size'   : 18}
		# matplotlib.rc('font', **font)
		
		all_x_axis_metric_vals  = [50, 100, 200, 500, 1000]
	
		
		curr_out_dir = f"{base_res_dir}/plots/{rq_name}"
		curr_res_dir = f"{base_res_dir}/{domain}/RQs/{rq_name}/plots"
		
		if rq_name == "RQ1_Model_Performance_At_Equal_Num_Retrieved":
			model_key = "model~ckpt~anc_n_e~graph_config"
			curr_x_axis_metric = "top_k_retvr"
			xlabel = "Number of Items Retrieved"
			model_plot_params = {
				"model=TFIDF~ckpt=None~anc_n_e=None~graph_config=None" : {"name": r"\textsc{TF-IDF}"},
				"model=01_D_BE_TRP_100~ckpt=best_wrt_dev~anc_n_e=None~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
				"model=03_D_BE_MATCH_100~ckpt=best_wrt_dev~anc_n_e=None~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
				"model=01_S_BE_TRP_100~ckpt=best_wrt_dev~anc_n_e=None~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
				"model=03_S_BE_MATCH_100~ckpt=best_wrt_dev~anc_n_e=None~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
				"model=00_6_20_BE~ckpt=best_wrt_dev~anc_n_e=None~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base}}"},
				"model=CUR~ckpt=None~anc_n_e=50~graph_config=None" : {"name": r"\textsc{annCUR}\textsubscript{50}"},
				"model=CUR~ckpt=None~anc_n_e=100~graph_config=None" : {"name": r"\textsc{annCUR}\textsubscript{100}"},
				"model=CUR~ckpt=None~anc_n_e=200~graph_config=None" : {"name": r"\textsc{annCUR}\textsubscript{200}"},
			}
			model_name_to_plots_params = {
				r"\textsc{TF-IDF}" : {"color": "lightseagreen"},
				r"\textsc{DE\textsubscript{base}}" : {"color": "gold"},
				r"\textsc{DE\textsubscript{base+ce}}" : {"color": "darkorange"},
				r"\textsc{DE\textsubscript{bert+ce}}" : {"color": "maroon"},
				
				r"\textsc{annCUR}\textsubscript{50}" : {"color": "yellowgreen"},
				r"\textsc{annCUR}\textsubscript{100}" : {"color": "limegreen"},
				r"\textsc{annCUR}\textsubscript{200}" : {"color": "darkgreen"},
			}
		elif rq_name == "RQ2_Model_Performance_At_Equal_Test_Cost":
			model_key = "model~ckpt~graph_config"
			curr_x_axis_metric = "cost"
			xlabel = "Inference Cost"
			
			model_plot_params = {
				"model=TFIDF~ckpt=None~graph_config=None" : {"name": r"\textsc{TF-IDF}"},
				"model=01_D_BE_TRP_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
				"model=03_D_BE_MATCH_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
				"model=01_S_BE_TRP_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}" },
				"model=03_S_BE_MATCH_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}" },
				"model=00_6_20_BE~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base}}" },
				"model=CUR~ckpt=None~graph_config=None" : {"name": r"\textsc{annCUR}"},
			}
			model_name_to_plots_params = {
				r"\textsc{TF-IDF}" : {"color": "lightseagreen"},
				r"\textsc{DE\textsubscript{base}}" : {"color": "gold"},
				r"\textsc{DE\textsubscript{base+ce}}" : {"color": "darkorange"},
				r"\textsc{DE\textsubscript{bert+ce}}" : {"color": "maroon"},
				
				r"\textsc{annCUR}" : {"color": "yellowgreen"},
			}
		else:
			raise NotImplementedError(f"RQ = {rq_name} not supported")
		
		all_y_vals = {}
		for topk in topk_vals:
			if curr_x_axis_metric == "cost":
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v > topk]
			else:
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v >= topk]
			
			file_name = f"{curr_res_dir}/nm_train={nm_train}~split_idx={0}~top_k={topk}~data_type={eval_data}.csv"
	
			if not os.path.isfile(file_name):
				LOGGER.info(f"File = {file_name} does not exist")
				continue
				
			y_vals = defaultdict(lambda : defaultdict(float))
			with open(file_name, "r") as fin:
				reader = csv.DictReader(fin)
				for ctr, row in enumerate(reader):
				
					curr_model = row[model_key]
					if curr_model not in model_plot_params: continue
					
					model_name = model_plot_params[curr_model]["name"]
					
					# LOGGER.info(f"Reading data for {model_name}")
					for x_val in x_axis_metric_vals:
						if f"{curr_x_axis_metric}={x_val}" not in row: continue
						
						curr_val = row[f"{curr_x_axis_metric}={x_val}"]
						curr_val = float(curr_val) if curr_val != "" else 0.
						
						prev_val = y_vals[model_name][f"{curr_x_axis_metric}={x_val}"]
						y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] = max(curr_val , prev_val)
			
			
			all_y_vals[topk] = y_vals
			
			LOGGER.info("Not plotting individual topk plots")
			continue
			# Now plot
			width = 0.1
			X = np.arange(len(x_axis_metric_vals))
			plt.clf()
			fig, ax1 = plt.subplots()
		
			for mctr, model_name in enumerate(model_name_to_plots_params):
				# Plot performance for current model for all domains
				
				Y = [y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] if f"{curr_x_axis_metric}={x_val}" in y_vals[model_name] else 0.
					 for x_val in x_axis_metric_vals]
				ax1.bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
				
			
			ax1.set_xticks(X + 0.5, x_axis_metric_vals)
			ax1.set_xlabel(xlabel)
			ax1.set_ylabel(f"Top-{topk}-Recall")
			ax1.legend()
			
			plt.tight_layout()
			
			out_file = f"{curr_out_dir}/domain={domain}/nm_train={nm_train}/topk={topk}.pdf"
			Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			
			plt.savefig(out_file,bbox_inches='tight')
			plt.close()
			
			
		
		############################ NOW CREATE COMBINED PLOTS FOR ALL TOPK_VALS #######################################
		
		plt.clf()
		
		if curr_x_axis_metric == "cost":
			num_bars = [sum(1 for r in all_x_axis_metric_vals if r > x) for x in topk_vals]
		else:
			num_bars = [sum(1 for r in all_x_axis_metric_vals if r >= x) for x in topk_vals]
			
		fig, axes = plt.subplots(1, len(topk_vals), figsize=(16, 3), gridspec_kw={'width_ratios': num_bars}, sharey=True )
		
		width = 0.175 if curr_x_axis_metric == "cost" else 0.12
		for topk_ctr, topk in enumerate(sorted(topk_vals)):
			if curr_x_axis_metric == "cost":
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v > topk]
			else:
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v >= topk]
				
			y_vals = all_y_vals[topk]
			X = np.arange(len(x_axis_metric_vals))
			for mctr, model_name in enumerate(model_name_to_plots_params):
				# Plot performance for current model for all domains
				
				Y = [y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] if f"{curr_x_axis_metric}={x_val}" in y_vals[model_name] else 0.
					 for x_val in x_axis_metric_vals]
				axes[topk_ctr].bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
				
			
			axes[topk_ctr].set_xticks(X + 0.5, x_axis_metric_vals)
			axes[topk_ctr].tick_params(axis='both', which='major', labelsize=20)
			
			# axes[topk_ctr].set_xlabel(xlabel)
			axes[topk_ctr].set_ylabel(r"Top-$k$-Recall", fontsize=24)
			axes[topk_ctr].set_title(r"$k$"+f"={topk}", fontsize=24)
		
			axes[topk_ctr].grid(axis='y', linestyle="--")
			axes[topk_ctr].set_axisbelow(True) # Push gridlines and axis below all other objects
			
		axes[0].set_xlabel(xlabel, fontsize=24)
		handles, labels = axes[0].get_legend_handles_labels()
		if curr_x_axis_metric == "cost":
			plt.legend(bbox_to_anchor=(0.4, 1.15), loc='lower left', ncol=len(handles), fontsize=17, bbox_transform=axes[0].transAxes, handletextpad=0.5, columnspacing=1)
			axes[0].xaxis.set_label_coords(2, -0.15)
		elif curr_x_axis_metric == "top_k_retvr":
			plt.legend(bbox_to_anchor=(-0.28, 1.15), loc='lower left', ncol=len(handles), fontsize=17, bbox_transform=axes[0].transAxes, handletextpad=0.5, columnspacing=1)
			axes[0].xaxis.set_label_coords(2, -0.15)
		else:
			raise NotImplementedError(f"curr_x_axis_metric  = {curr_x_axis_metric} not supported")
		
		plt.ylim(0,100)
		plt.subplots_adjust(wspace=0.1, hspace=0)
		plt.tight_layout()
		
		for ax in fig.get_axes(): ax.label_outer() # Label only outermost axes
		
		rq_short = "rq_2" if curr_x_axis_metric == "cost" else "rq_1"
		out_file = f"{curr_out_dir}/domain={domain}/nm_train={nm_train}/{rq_short}_domain={domain}_nm_train={nm_train}_recall_vs_{curr_x_axis_metric}_topk={'_'.join([str(x) for x in topk_vals])}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		
		plt.savefig(out_file,bbox_inches='tight')
		plt.close()
		
					
	except Exception as e:
		embed()
		raise e


def plot_rq_5_6_performance_vs_topk_retrieved_or_cost_for_CE_only_baselines(base_res_dir, nm_train, domain, rq_name, topk_vals, eval_data="test"):
	
	try:
		
		all_x_axis_metric_vals  = [50, 100, 200, 500, 1000]

		
		if rq_name == "RQ5_Model_Performance_At_Equal_Test_Cost_CE_Baselines":
			curr_out_dir = f"{base_res_dir}/plots/{rq_name}"
			curr_res_dir = f"{base_res_dir}/{domain}/RQs/RQ2_Model_Performance_At_Equal_Test_Cost/plots"
			model_key = "model~ckpt~graph_config"
			curr_x_axis_metric = "cost"
			xlabel = "Inference Cost"
			
			model_plot_params = {
				"model=CUR~ckpt=None~graph_config=None" : {"name": r"\textsc{CUR}"},
				"model=Fixed_Anc_Ent~ckpt=None~graph_config=None" : {"name": "Fixed_Item"},
				"model=Fixed_Anc_Ent_CUR~ckpt=None~graph_config=None" : {"name": "ItemCUR"}
			}
			model_name_to_plots_params = {
				"Fixed_Item" : {"color": "darkgreen"},
				"ItemCUR" : {"color": "forestgreen"},
				"CUR" : {"color": "yellowgreen"},
			}
		elif rq_name == "RQ6_Model_Performance_At_Equal_Num_Retrieved_CE_Baselines":
			curr_out_dir = f"{base_res_dir}/plots/{rq_name}"
			curr_res_dir = f"{base_res_dir}/{domain}/RQs/RQ1_Model_Performance_At_Equal_Num_Retrieved/plots"
			
			model_key = "model~ckpt~anc_n_e~graph_config"
			curr_x_axis_metric = "top_k_retvr"
			xlabel = "Number of Items Retrieved"
			
		
		
			model_plot_params = {
				"model=CUR~ckpt=None~anc_n_e=100~graph_config=None" : {"name": r"\textsc{annCUR}\textsubscript{100}"},
				"model=CUR~ckpt=None~anc_n_e=200~graph_config=None" : {"name": r"\textsc{annCUR}\textsubscript{200}"},
				
				"model=Fixed_Anc_Ent_CUR~ckpt=None~anc_n_e=100~graph_config=None" : {"name": r"\textsc{itemCUR}\textsubscript{100}"},
				"model=Fixed_Anc_Ent_CUR~ckpt=None~anc_n_e=200~graph_config=None" : {"name": r"\textsc{itemCUR}\textsubscript{200}"},
				
				"model=Fixed_Anc_Ent~ckpt=None~anc_n_e=200~graph_config=None" : {"name": r"\textsc{fixedITEM}"},
			}
			
			model_name_to_plots_params = {
				# "Fixed_Item" : {"color": "orange"},
				r"\textsc{fixedITEM}" : {"color": "darkblue"},
				r"\textsc{itemCUR}\textsubscript{100}" : {"color": "turquoise"},
				r"\textsc{itemCUR}\textsubscript{200}" : {"color": "teal"},
				r"\textsc{annCUR}\textsubscript{100}" : {"color": "yellowgreen"},
				r"\textsc{annCUR}\textsubscript{200}"	 : {"color": "forestgreen"},
			}
		else:
			raise NotImplementedError(f"RQ = {rq_name} not supported")
		
		all_y_vals = {}
		for topk in topk_vals:
			if curr_x_axis_metric == "cost":
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v > topk]
			else:
				x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v >= topk]
			
			file_name = f"{curr_res_dir}/nm_train={nm_train}~split_idx={0}~top_k={topk}~data_type={eval_data}.csv"
	
			if not os.path.isfile(file_name):
				LOGGER.info(f"File = {file_name} does not exist")
				continue
				
			y_vals = defaultdict(lambda : defaultdict(float))
			with open(file_name, "r") as fin:
				reader = csv.DictReader(fin)
				for ctr, row in enumerate(reader):
				
					curr_model = row[model_key]
					if curr_model not in model_plot_params: continue
					
					model_name = model_plot_params[curr_model]["name"]
					
					# LOGGER.info(f"Reading data for {model_name}")
					for x_val in x_axis_metric_vals:
						if f"{curr_x_axis_metric}={x_val}" not in row: continue
						
						curr_val = row[f"{curr_x_axis_metric}={x_val}"]
						curr_val = float(curr_val) if curr_val != "" else 0.
						
						prev_val = y_vals[model_name][f"{curr_x_axis_metric}={x_val}"]
						y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] = max(curr_val , prev_val)
			
			
			all_y_vals[topk] = y_vals
			# Now plot
			width = 0.3 if curr_x_axis_metric == "cost" else 0.18
			X = np.arange(len(x_axis_metric_vals))
			plt.clf()
			fig, ax1 = plt.subplots(figsize=(6,3.5))
		
		
			for mctr, model_name in enumerate(model_name_to_plots_params):
				# Plot performance for current model for all domains
				
				Y = [y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] if f"{curr_x_axis_metric}={x_val}" in y_vals[model_name] else 0.
					 for x_val in x_axis_metric_vals]
				ax1.bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
				
			
			ax1.set_xticks(X + 0.5, x_axis_metric_vals, fontsize=24)
			ax1.set_xlabel(xlabel, fontsize=24)
			ax1.set_ylabel(f"Top-{topk}-Recall", fontsize=24)
			plt.setp(ax1.get_yticklabels(), fontsize=24)
			ax1.grid(axis='y', linestyle="--")
			ax1.set_axisbelow(True) # Push gridlines and axis below all other objects
			
			# ax1.legend(prop={'size': 24})
			
			handles, labels = ax1.get_legend_handles_labels()
			plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.08, 1.002), loc='lower left', ncol=int(len(handles)/2),
					   fontsize=16, bbox_transform=ax1.transAxes)
			
			plt.tight_layout()
			
			out_file = f"{curr_out_dir}/domain={domain}/nm_train={nm_train}/rq_6_domain={domain}_nm_train={nm_train}_recall_vs_num_retrieved_topk={topk}.pdf"
			Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			
			plt.savefig(out_file,bbox_inches='tight')
			plt.close()
			
			
		
		
		return
		# plt.clf()
		#
		# if curr_x_axis_metric == "cost":
		# 	num_bars = [sum(1 for r in all_x_axis_metric_vals if r > x) for x in topk_vals]
		# else:
		# 	num_bars = [sum(1 for r in all_x_axis_metric_vals if r >= x) for x in topk_vals]
		#
		# fig, axes = plt.subplots(1, len(topk_vals), figsize=(16, 6), gridspec_kw={'width_ratios': num_bars}, sharey=True )
		#
		# width = 0.3 if curr_x_axis_metric == "cost" else 0.18
		# for topk_ctr, topk in enumerate(sorted(topk_vals)):
		# 	if curr_x_axis_metric == "cost":
		# 		x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v > topk]
		# 	else:
		# 		x_axis_metric_vals = [v for v in all_x_axis_metric_vals if v >= topk]
		#
		# 	y_vals = all_y_vals[topk]
		# 	X = np.arange(len(x_axis_metric_vals))
		# 	for mctr, model_name in enumerate(model_name_to_plots_params):
		# 		# Plot performance for current model for all domains
		#
		# 		Y = [y_vals[model_name][f"{curr_x_axis_metric}={x_val}"] if f"{curr_x_axis_metric}={x_val}" in y_vals[model_name] else 0.
		# 			 for x_val in x_axis_metric_vals]
		# 		axes[topk_ctr].bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
		#
		#
		# 	axes[topk_ctr].set_xticks(X + 0.5, x_axis_metric_vals, fontsize=12)
		# 	axes[topk_ctr].tick_params(axis='both', which='major', labelsize=12)
		#
		# 	# axes[topk_ctr].set_xlabel(xlabel)
		# 	# axes[topk_ctr].set_ylabel(f"Top-{topk}-Recall", fontsize=16)
		# 	axes[topk_ctr].set_ylabel(f"Top-k-Recall", fontsize=16)
		# 	axes[topk_ctr].set_title(f"k={topk}")
		# 	# axes[topk_ctr].legend()
		#
		# axes[0].set_xlabel(xlabel, fontsize=16)
		# handles, labels = axes[0].get_legend_handles_labels()
		# if curr_x_axis_metric == "cost":
		# 	plt.legend(bbox_to_anchor=(0.4, 1.05), loc='lower left', ncol=len(handles), fontsize=15, bbox_transform=axes[0].transAxes)
		# 	axes[0].xaxis.set_label_coords(1.5, -0.08)
		# elif curr_x_axis_metric == "top_k_retvr":
		# 	plt.legend(bbox_to_anchor=(-0.02, 1.05), loc='lower left', ncol=len(handles), fontsize=15, bbox_transform=axes[0].transAxes)
		# 	axes[0].xaxis.set_label_coords(2, -0.08)
		# else:
		# 	raise NotImplementedError(f"curr_x_axis_metric  = {curr_x_axis_metric} not supported")
		#
		#
		#
		#
		# plt.ylim(0,100)
		# plt.subplots_adjust(wspace=0.1, hspace=0)
		# plt.tight_layout()
		#
		# for ax in fig.get_axes(): ax.label_outer() # Label only outermost axes
		#
		# rq_short = "rq_5" if curr_x_axis_metric == "cost" else "rq_6"
		# out_file = f"{curr_out_dir}/domain={domain}/nm_train={nm_train}/{rq_short}_domain={domain}_nm_train={nm_train}_recall_vs_{curr_x_axis_metric}_topk={'_'.join([str(x) for x in topk_vals])}.pdf"
		# Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		#
		# plt.savefig(out_file,bbox_inches='tight')
		# plt.close()
		
		
	except Exception as e:
		# embed()
		raise e


def plot_rq_3_performance_vs_domains_size(res_dir, topk, cost, nm_train):
	
	"""
	X - domains
	Y - Plot for top-{topk}-recall-at-cost-{cost} for all 5 domains for n-train={nm_train} - do so for all relevant models
	:param res_dir:
	:return:
	"""

	try:

		domains = {
			"pro_wrestling":"Pro-Wrestling",
			"yugioh":"YuGiOh",
			"star_trek":"Star-Trek",
			"doctor_who":"Doctor-Who",
			"military":"Military"
		}
		
		
		model_plot_params = {
			"model=01_D_BE_TRP_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
			"model=03_D_BE_MATCH_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
			"model=01_S_BE_TRP_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
			"model=03_S_BE_MATCH_100~ckpt=best_wrt_dev~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
			"model=00_6_20_BE~ckpt=best_wrt_dev~graph_config=None" : {"name": r'\textsc{DE\textsubscript{base}'},
			"model=CUR~ckpt=None~graph_config=None" : {"name": r"\textsc{annCUR}"},
		}
		model_name_to_plots_params = {
			r'\textsc{DE\textsubscript{base}' : {"color": "gold"},
			r"\textsc{DE\textsubscript{base+ce}}" : {"color": "darkorange"},
			r"\textsc{DE\textsubscript{bert+ce}}" : {"color": "maroon"},
			r"\textsc{annCUR}" : {"color": "yellowgreen"},
		}
	
		y_vals = defaultdict(lambda : defaultdict(float))
		
		for domain in domains:
			file_name = f"{res_dir}/{domain}/RQs/RQ2_Model_Performance_At_Equal_Test_Cost/plots/nm_train={nm_train}~split_idx=0~top_k={topk}~data_type=test.csv"
			if not os.path.isfile(file_name):
				LOGGER.info(f"File = {file_name} does not exist")
				continue
				
			with open(file_name, "r") as fin:
				
				reader = csv.DictReader(fin)
				for ctr, row in enumerate(reader):
					
					
					curr_model = row["model~ckpt~graph_config"]
					if curr_model not in model_plot_params: continue
					
					# assert domain not in y_vals[row["model~ckpt"]], f"Multiple values for the same model = {row['model~ckpt']} and domain={domain}"
					
					model_name = model_plot_params[curr_model]["name"]
					curr_val = float(row[f"cost={cost}"]) if row[f"cost={cost}"] != "" else 0.
					y_vals[model_name][domain] = max(curr_val, y_vals[model_name][domain])
	
	
		
		# Now plot
		width = 0.20
		X = np.arange(len(domains))
		plt.clf()
		fig, ax1 = plt.subplots(figsize=(6,3.5))

		ax2 = ax1.twinx()
		
		sec_y_color = "deepskyblue"
		sec_y_color = "royalblue"
		ax2.plot(X + 0.5, [NUM_ENTS[domain] for domain in domains], "-*", color=sec_y_color)
		ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.e'))
		ax2.set_ylabel(f"Number of items", fontsize=16)
		plt.setp(ax2.get_yticklabels(), fontsize=12, color=sec_y_color)
		ax2.yaxis.label.set_color(sec_y_color)
		
		
		for mctr, model_name in enumerate(model_name_to_plots_params):
			# Plot performance for current model for all domains
			
			Y = [y_vals[model_name][domain] if domain in y_vals[model_name] else 0. for domain in domains]
			ax1.bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
			
		ax1.set_xticks(X + 0.5, domains.values(), fontsize=13)
		ax1.set_xlabel("Item Domains", fontsize=16)
		ax1.set_ylabel(f"Top-{topk}-Recall@Cost={cost}", fontsize=16)
		ax1.grid(axis='y', linestyle="--")
		ax1.set_axisbelow(True) # Push gridlines and axis below all other objects
		
		# ax1.legend(prop={'size': 14})
		handles, labels = ax1.get_legend_handles_labels()
		plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0.04, 1.002), loc='lower left', ncol=int(len(handles)/2),
				   fontsize=16, bbox_transform=ax1.transAxes)
		
		plt.setp(ax1.get_yticklabels(), fontsize=14)
		plt.tight_layout()
		
		out_file = f"{res_dir}/plots/RQ3_performance_vs_domain_size/nm_train={nm_train}_topk={topk}_cost={cost}/rq_3_perf_w_domain_size_nm_train={nm_train}_topk={topk}_cost={cost}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		
		plt.savefig(out_file,bbox_inches='tight')
		plt.close()
		
	except Exception as e:
		embed()
		raise e


def plot_rq_4_performance_vs_train_data_size(res_dir, topk, cost, domain_vals, eval_data):
	
	
	"""
	X - Size of training data
	Y - Plot for top-{topk}-recall-at-cost-{cost} for given domain as we vary number of mentions used for training.  - do so for all relevant models

	:param res_dir:
	:return:
	"""

	try:
		
		all_y_vals = {}
		ckpt_type = "best_wrt_dev" if eval_data == "test" else "eoe"
		model_plot_params = {
			f"model=01_D_BE_TRP_100~ckpt={ckpt_type}~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}" },
			f"model=03_D_BE_MATCH_100~ckpt={ckpt_type}~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base+ce}}"  },
			f"model=01_S_BE_TRP_100~ckpt={ckpt_type}~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
			f"model=03_S_BE_MATCH_100~ckpt={ckpt_type}~graph_config=None" : {"name": r"\textsc{DE\textsubscript{bert+ce}}"},
			f"model=00_6_20_BE~ckpt={ckpt_type}~graph_config=None" : {"name": r"\textsc{DE\textsubscript{base}}"},
			"model=CUR~ckpt=None~graph_config=None" : {"name": r"\textsc{annCUR}"},
		}
		model_name_to_plots_params = {
			r"\textsc{DE\textsubscript{base}}" : {"color": "gold"},
			r"\textsc{DE\textsubscript{base+ce}}" : {"color": "darkorange"},
			r"\textsc{DE\textsubscript{bert+ce}}" : {"color": "maroon"},
			r"\textsc{annCUR}" : {"color": "yellowgreen"},
		}
		
		for domain in domain_vals:
			nm_train_vals = [100, 500, 2000] if domain != "pro_wrestling" else [100, 500, 1000]
			
			y_vals = defaultdict(lambda : defaultdict(float))
			
			for nm_train in nm_train_vals:
				file_name = f"{res_dir}/{domain}/RQs/RQ2_Model_Performance_At_Equal_Test_Cost/plots/nm_train={nm_train}~split_idx=0~top_k={topk}~data_type={eval_data}.csv"
				if not os.path.isfile(file_name):
					LOGGER.info(f"File = {file_name} does not exist")
					continue
					
				with open(file_name, "r") as fin:
					
					reader = csv.DictReader(fin)
					for ctr, row in enumerate(reader):
						
						
						curr_model = row["model~ckpt~graph_config"]
						if curr_model not in model_plot_params: continue
						
						model_name = model_plot_params[curr_model]["name"]
						curr_val = float(row[f"cost={cost}"]) if row[f"cost={cost}"] != "" else 0.
						y_vals[model_name][nm_train] = max(curr_val, y_vals[model_name][nm_train])
		
			
			all_y_vals[domain] = y_vals
			
			# Now plot
	
			continue
			width = 0.20
			X = np.arange(len(nm_train_vals))
			plt.clf()
			fig, ax = plt.subplots()
			for mctr, model_name in enumerate(model_name_to_plots_params):
				# Plot performance for current model for all nm_train_vals
				
				Y = [y_vals[model_name][nm_train] if nm_train in y_vals[model_name] else 0. for nm_train in nm_train_vals]
				ax.bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
				
			
			plt.xticks(X + 0.5, nm_train_vals, fontsize=20)
			plt.xlabel("Size of training data", fontsize=20)
			plt.ylabel(f"Top-{topk}-Recall@Cost={cost}", fontsize=20)
			
			plt.setp(ax.get_yticklabels(), fontsize=16)
			
			plt.legend(prop={'size': 20}, ncol=2)
			plt.legend(bbox_to_anchor=(-0.002, 1.002), loc='lower left', ncol=2,
					   fontsize=20, bbox_transform=ax.transAxes)
			# out_file = f"{res_dir}/plots/RQ4_performance_vs_train_data_size/{eval_data}/domain={domain}/rq_4_perf_vs_nm_train_domain={domain}_topk={topk}_cost={cost}.pdf"
			out_file = f"{res_dir}/plots/RQ4_performance_vs_train_data_size/{eval_data}/rq_4_perf_vs_nm_train_domain={domain}_topk={topk}_cost={cost}.pdf"
			Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
			
			plt.savefig(out_file,bbox_inches='tight')
			plt.close()
			
		
		
		
		#################################### NOW PLOT ON ALL DOMAINS IN A SINGLE PLOT ########################################
		
		plt.clf()
		
		fig, axes = plt.subplots(1, len(domain_vals), figsize=(4*len(domain_vals), 3), sharey=True )
		
		width = 0.2
		for domain_ctr, domain in enumerate(domain_vals):
			nm_train_vals = [100, 500, 2000] if domain != "pro_wrestling" else [100, 500, 1000]
			
			y_vals = all_y_vals[domain]
			X = np.arange(len(nm_train_vals))
			for mctr, model_name in enumerate(model_name_to_plots_params):
				# Plot performance for current model for all domains
				
				Y = [y_vals[model_name][nm_train] if nm_train in y_vals[model_name] else 0.
					 for nm_train in nm_train_vals]
				axes[domain_ctr].bar(X + (mctr + 1)*width, Y, width, color=model_name_to_plots_params[model_name]["color"], label=model_name)
				
			
			axes[domain_ctr].set_xticks(X + 0.5, nm_train_vals)
			axes[domain_ctr].tick_params(axis='both', which='major', labelsize=24)
			
			
			axes[domain_ctr].set_ylabel(f"Top-{topk}-Recall@Cost={cost}", fontsize=24)
			axes[domain_ctr].set_title(f"{domain_vals[domain]}", fontsize=24)
			axes[domain_ctr].grid(axis='y', linestyle="--")
			axes[domain_ctr].set_axisbelow(True) # Push gridlines and axis below all other objects
			
		
		axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))
		axes[0].set_xlabel(r"Number of Queries in Indexing/Training Data ($ \vert \mathcal{Q}\textsubscript{train} \vert $) ", fontsize=24)
		handles, labels = axes[0].get_legend_handles_labels()
		
		
		plt.setp(axes[0].get_yticklabels(), fontsize=16)
		if len(domain_vals) == 5:
			plt.legend(bbox_to_anchor=(0.8, 1.1), loc='lower left', ncol=len(handles), fontsize=24, bbox_transform=axes[0].transAxes)
			axes[0].xaxis.set_label_coords(2.75, -0.18)
		elif len(domain_vals) == 3:
			plt.legend(bbox_to_anchor=(0.8, 1.1), loc='lower left', ncol=len(handles), fontsize=24, bbox_transform=axes[0].transAxes)
		else:
			plt.legend(handles=handles, labels=labels)
			
		plt.ylim(0,100)
		plt.setp(axes[0].get_yticklabels(), fontsize=24)
		for ax in fig.get_axes(): ax.label_outer() # Label only outermost axes
		
		plt.subplots_adjust(wspace=0.1, hspace=0)
		plt.tight_layout()
		
		out_file = f"{res_dir}/plots/RQ4_performance_vs_train_data_size/{eval_data}/rq_4_perf_vs_nm_train_topk={topk}_cost={cost}_domains={'_'.join(domain_vals)}.pdf"
		Path(os.path.dirname(out_file)).mkdir(exist_ok=True, parents=True)
		
		plt.savefig(out_file,bbox_inches='tight')
		plt.close()
		
		
		
	except Exception as e:
		embed()
		raise e


def plot_rq7_heatmaps(data_file, method_vals, crossenc_type, out_dir):
	
	try:
		colors = [('firebrick', 'lightsalmon'),('green', 'yellowgreen'),
		('navy', 'skyblue'), ('olive', 'y'),
		('sienna', 'tan'), ('darkviolet', 'orchid'),
		('darkorange', 'gold'), ('deeppink', 'violet'),
		('deepskyblue', 'lightskyblue'), ('gray', 'silver')]
		LOGGER.info("Now plotting results")
		
		with open(data_file, "r") as fin:
			eval_res = json.load(fp=fin)
	
		n_ment_anchors_vals = eval_res["other_args"]["n_ment_anchors_vals"]
		n_ent_anchors_vals = eval_res["other_args"]["n_ent_anchors_vals"]
		top_k_vals = eval_res["other_args"]["top_k_vals"]
		top_k_retvr_vals = eval_res["other_args"]["top_k_retr_vals"]
		
		data_name =  eval_res["other_args"]["arg_dict"]["data_name"]
		n_ent = NUM_ENTS[data_name]
		
		top_k_vals = [10]
		top_k_retvr_vals = [500]
		n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v < n_ent]
		
		
		n_ment_anchors_vals = sorted(n_ment_anchors_vals, reverse=True)
		
		n_ent_anchors_vals = [v for v in n_ent_anchors_vals if v <= 1000]
		n_ment_anchors_vals = [v for v in n_ment_anchors_vals if v <= 1000]
		
		############################# NOW VISUALIZE RESULTS AS A FUNCTION OF N_ANCHORS ####################################
		metrics = {
			f"exact_vs_reranked_approx_retvr~common_frac_mean": "prec_at_k",
			f"approx_error_relative": "approx_error",
		}
	
		mtype_vals = ["non_anchor"]
		for mtype, curr_method, top_k, top_k_retvr, metric in itertools.product(mtype_vals, method_vals, top_k_vals, top_k_retvr_vals, metrics):
			if top_k > top_k_retvr: continue
			val_matrix = []
			# Build matrix for given topk value with varying number of anchor mentions and anchor entities
			try:
				for n_ment_anchors in n_ment_anchors_vals:
					curr_config_res = [100*eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"][f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}"][mtype][metric]
									   if f"anc_n_m={n_ment_anchors}~anc_n_e={n_ent_anchors}" in eval_res[curr_method][f"top_k={top_k}"][f"k_retvr={top_k_retvr}"]
									   else 0.
									   for n_ent_anchors in n_ent_anchors_vals]
					val_matrix += [curr_config_res]
			except KeyError as e:
				LOGGER.info(f"Key-error = {e} for mtype = {mtype}, curr_method={curr_method}, top_k={top_k}, top_k_retvr={top_k_retvr}, metric={metric}")
				continue
	
			val_matrix = np.array(val_matrix, dtype=np.float64)
			curr_res_dir = f"{out_dir}/plots_{mtype}/k={top_k}/k_retr={top_k_retvr}_{curr_method}"
			if metrics[metric] == "approx_error":
				out_fname = f"{curr_res_dir}/rq_7_2_arch_change_{data_name}_{metrics[metric]}_heatmap_top_{top_k}_at_{top_k_retvr}_{curr_method}_{crossenc_type}.pdf"
			else:
				out_fname = f"{curr_res_dir}/rq_7_1_arch_change_{data_name}_{metrics[metric]}_heatmap_top_{top_k}_at_{top_k_retvr}_{curr_method}_{crossenc_type}.pdf"
			plot_heat_map(
				val_matrix=val_matrix,
				row_vals=n_ment_anchors_vals,
				col_vals=n_ent_anchors_vals,
				out_fname=out_fname,
				metric=metrics[metric]
			)
	

	except Exception as e:
		embed()
		raise e


def plot_heat_map(val_matrix, row_vals, col_vals, out_fname, metric):
	"""
	Plot a heat map using give matrix and add x-/y-ticks and title
	:param val_matrix: Matrix for plotting heat map
	:param row_vals: y-ticks
	:param col_vals: x-ticks
	:param metric:
	:param top_k:
	:param curr_res_dir:
	:return:
	"""
	try:
		
		# for cmap_str, cmaps in [("Blues", cm.Blues) , ("YlOrBr", cm.YlOrBr) , ("OrRd", cm.OrRd) , ("PuRd", cm.PuRd) , ("YlGn", cm.YlGn) , ("ocean", cm.ocean) , ("winter", cm.winter) , ("Wistia", cm.Wistia)]:
		for cmap_str, cmaps in [("Blues", cm.Blues)]:
			plt.clf()
			# try:
			# 	if np.max(val_matrix) > 100:
			# 		# fig, ax = plt.subplots(figsize=(12,12))
			# 		fig, ax = plt.subplots(figsize=(8,8))
			# 	else:
			# 		fig, ax = plt.subplots(figsize=(8,8))
			# except:
			fig, ax = plt.subplots(figsize=(8,8))
			
			
			if metric == "prec_at_k":
				norm = plt.Normalize(vmin=-20, vmax=100, clip=True)
			elif metric == "approx_error":
				temp = copy.deepcopy(val_matrix)
				n = val_matrix.shape[0]
				for i in range(n):
					temp[i,n-i-1] = 0
				norm = plt.Normalize(vmin=0, vmax=np.max(temp)*1.2, clip=True)
			else:
				norm = plt.Normalize()
				
			im = ax.imshow(val_matrix, cmap=cmaps, norm=norm)
			
			# im = ax.imshow(np.log(val_matrix), cmap=cmaps, norm=norm)
			# im = ax.imshow(val_matrix, cmap=cmaps)
			
			# We want to show all ticks...
			ax.set_xticks(np.arange(len(col_vals)))
			ax.set_yticks(np.arange(len(row_vals)))
			# ... and label them with the respective list entries
			ax.set_xticklabels(col_vals)
			ax.set_yticklabels(row_vals)
		
			# Rotate the tick labels and set their alignment.
			plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
					 rotation_mode="anchor", fontsize=40)
			
			plt.setp(ax.get_yticklabels(), rotation=0, fontsize=40)
		
			# Loop over data dimensions and create text annotations.
			for i in range(len(row_vals)):
				for j in range(len(col_vals)):
					fontsize = 40 if val_matrix[i,j] < 100 else 32
					ax.text(j, i, "{:.1f}".format(val_matrix[i, j]),
							ha="center", va="center", color="w", fontsize=fontsize)
		 
			# ax.set_title(f"{metric} for topk={top_k}" if title is None else title)
			ax.set_xlabel("Number of anchor items", fontsize=40)
			ax.set_ylabel("Number of anchor queries", fontsize=40)
			fig.tight_layout()
			
			# plt.savefig(f"{curr_res_dir}/{metric}_{top_k}_{cmap_str}.pdf")
			Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
			plt.savefig(f"{out_fname}")
			plt.close()
	except Exception as e:
		embed()
		raise e


def plot_rq_0_score_distribution(out_dir, bins, plot_per_row):
	
	data_name = "yugioh"
	num_rows = 100
	mat_files = {
		"bienc" : {
			"color":"gold",
			"label":r"\textsc{DE}",
			"file": f"../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/score_mats_model-3-12039.0-2.17.ckpt/{data_name}/ment_to_ent_scores_{3374}x{10031}.pkl",
		},
		"e_crossenc_6_256" 		: {
			"color":"lightcoral",
			"label":r"\textsc{[emb]-CE}",
			"file": f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt/{data_name}/ment_to_ent_scores_n_m_3374_n_e_10031_all_layers_False.pkl"
		},
		"cls_crossenc_6_49_rep" : {
			"color":"maroon",
			"label":r"\textsc{[cls]-CE}",
			"file": f"../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt/{data_name}/ment_to_ent_scores_n_m_3374_n_e_10031_all_layers_False.pkl",
		},
	}
	
	# for curr_train_setting in mat_files:
	# 	plt.clf()
	# 	fig, ax = plt.subplots(figsize=(10,8))
	#
	# 	LOGGER.info(f"Processing dir {curr_train_setting}")
	# 	matrix_file = mat_files[curr_train_setting]["file"]
	# 	if not os.path.exists(matrix_file):
	# 		LOGGER.info(f"Skipping {matrix_file} as matrix file does not exist\n")
	# 		continue
	#
	# 	with open(matrix_file, "rb") as fin:
	# 		dump_dict = pickle.load(fin)
	# 		val_matrix = dump_dict["ment_to_ent_scores"]
	# 		if isinstance(val_matrix, torch.Tensor):
	# 			val_matrix = val_matrix.cpu().numpy()
	#
	# 	rank = np.linalg.matrix_rank(val_matrix)
	# 	LOGGER.info(f"Rank of matrix = {rank}")
	#
	# 	if plot_per_row:
	# 		val_matrix = val_matrix[:num_rows]
	# 		for row in val_matrix:
	# 			sns.distplot(row,
	# 						 hist = True, kde = True,
	# 						 kde_kws = {'shade': True, 'linewidth': 2},
	# 						 bins = bins,
	# 						 color=mat_files[curr_train_setting]["color"],
	# 						 ax=ax)
	#
	# 	else:
	# 		val_matrix = val_matrix[:num_rows]
	# 		val_matrix = val_matrix.reshape(-1)
	# 		val_matrix = val_matrix - np.mean(val_matrix)
	#
	# 		sns.distplot(val_matrix,
	# 					 hist = True, kde = True,
	# 					 kde_kws = {'shade': True, 'linewidth': 2},
	# 					 bins = bins,
	# 					 color=mat_files[curr_train_setting]["color"],
	# 					 ax=ax)
	# 		ax.set_xlim(-15,15)
	#
	# 	ax.set_xlabel("Query-Item Score", fontsize=20)
	# 	ax.set_ylabel("Score Density", fontsize=20)
	#
	# 	ax.tick_params(axis='both', which='major', labelsize=16)
	# 	fig.tight_layout()
	#
	# 	out_filename = f"{out_dir}/{data_name}/score_dist_n={num_rows}_{curr_train_setting}_bins={bins}.pdf"
	# 	Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
	# 	plt.savefig(out_filename)
	# 	plt.close()
	#
	
	#################################### PLOT ALL ON A SINGLE PLOT #####################################################
	plt.clf()
	fig, ax = plt.subplots(figsize=(10,8))
	
	for curr_train_setting in mat_files:
		
		LOGGER.info(f"Processing dir {curr_train_setting}")
		matrix_file = mat_files[curr_train_setting]["file"]
		if not os.path.exists(matrix_file):
			LOGGER.info(f"Skipping {matrix_file} as matrix file does not exist\n")
			continue
	
		with open(matrix_file, "rb") as fin:
			dump_dict = pickle.load(fin)
			val_matrix = dump_dict["ment_to_ent_scores"]
			if isinstance(val_matrix, torch.Tensor):
				val_matrix = val_matrix.cpu().numpy()
		
		
		if plot_per_row:
			val_matrix = val_matrix[:num_rows]
			for row in val_matrix:
				sns.distplot(row,
							 hist = True, kde = True,
							 kde_kws = {'shade': True, 'linewidth': 2,},
							 bins = bins,
							 color=mat_files[curr_train_setting]["color"],
							 ax=ax
							 )
		else:
			val_matrix = val_matrix[:num_rows].reshape(-1)
			val_matrix = val_matrix - np.mean(val_matrix)
		
			sns.distplot(val_matrix,
						 hist = True, kde = True,
						 kde_kws = {'shade': True, 'linewidth': 2,},
						 bins = bins,
						 color=mat_files[curr_train_setting]["color"],
						 ax=ax, label=mat_files[curr_train_setting]["label"])
			ax.set_xlim(-15,15)
		
	ax.set_xlabel("Query-Item Score", fontsize=50)
	ax.set_ylabel("Score Density", fontsize=50)
	ax.tick_params(axis='both', which='major', labelsize=40)
	ax.tick_params(axis='x', which='major', width=2, length=20, direction="inout")
	ax.grid(axis="x", linestyle="--", zorder=0.5)
	# ax.set_axisbelow(True) # Push gridlines and axis below all other objects
	plt.legend(prop={'size': 30})
	fig.tight_layout()
	
	out_filename = f"{out_dir}/{data_name}/score_dist_n={num_rows}_joint_bins={bins}.pdf"
	Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
	plt.savefig(out_filename, bbox_inches='tight')
	
	plt.yscale('log')
	out_filename = f"{out_dir}/{data_name}/score_dist_n={num_rows}_joint_log_scale_bins={bins}.pdf"
	Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
	plt.savefig(out_filename, bbox_inches='tight')

	
	
	# for ylim in [0.001, 0.01, 0.1]:
	for ylim in [0.001]:
		plt.yscale('linear')
		plt.gca().set_ylim(bottom=ylim)
		out_filename = f"{out_dir}/{data_name}/score_dist_n={num_rows}_joint_linear_scale_ylim_{ylim}_bins={bins}.pdf"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename, bbox_inches='tight')
		
		plt.yscale('log')
		plt.gca().set_ylim(bottom=ylim)
		out_filename = f"{out_dir}/{data_name}/score_dist_n={num_rows}_joint_log_scale_ylim_{ylim}_bins={bins}.pdf"
		Path(os.path.dirname(out_filename)).mkdir(exist_ok=True, parents=True)
		plt.savefig(out_filename, bbox_inches='tight')
		
	plt.close()
	

def plot_heatmap_for_cur_diagram(res_dir):
	
	try:
		extension = "pdf"
		scale = 10
		n_rows = int(80/scale)
		n_cols = int(200/scale)
		n_anchor_cols = int(50/scale)
		n_anchor_rows = int(80/scale)
		
		anchor_cols = int((n_cols - n_anchor_cols)/2)  + np.arange(n_anchor_cols)
		anchor_rows = np.arange(n_anchor_rows)
		
		n_test_rows = int(10/scale)
		train_cmap = cm.Blues
		test_cmap = cm.YlGn
		U_cmap = cm.Reds
		itemEmbed_cmap = cm.RdPu
		
		"""
		100x200 = 100 x 50      50 x 70      70 x 200
		"""
		# for cmap_str, cmaps in [("Blues", cm.Blues) , ("YlOrBr", cm.YlOrBr) , ("OrRd", cm.OrRd) , ("PuRd", cm.PuRd) , ("YlGn", cm.YlGn) , ("ocean", cm.ocean) , ("winter", cm.winter) , ("Wistia", cm.Wistia)]:
		# for cmap_str, cmaps in [("Blues", cm.Blues)]:
			
		rng = np.random.default_rng(seed=0)
		
		M = [ np.abs(rng.normal(loc=0, scale=1, size=n_cols)) for _ in range(n_rows) ]
		M = np.array([ M[row_iter]/np.linalg.norm(M[row_iter], ord=1) for row_iter in range(n_rows) ])
		
		vmax = .15
		################### Plot entire M Matrix
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_cols,n_rows))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(M, cmap=train_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_M.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		
		################### Plot entire C - a subset of columns of  M
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_anchor_cols,n_rows))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(M[:, anchor_cols], cmap=train_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_C.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		################### Plot entire R - a subset of columns of  M
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_cols,n_anchor_rows))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(M[anchor_rows, :], cmap=train_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_R.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		
		
		
		
		
		
		
		
		
		
		
		M_test = [ np.abs(rng.normal(loc=0, scale=1, size=n_cols)) for _ in range(n_test_rows) ]
		M_test = np.array([ M_test[row_iter]/np.linalg.norm(M_test[row_iter], ord=1) for row_iter in range(n_test_rows) ])
		
		################### Plot entire M_test Matrix
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_cols,n_rows))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(M_test, cmap=test_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_M_test.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		
		
		
		################### Plot entire C_test - a subset of columns of  M_test
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_anchor_cols,n_rows))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(M_test[:, anchor_cols], cmap=test_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_C_test.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		
		
		
		
		
		
		U = [ np.abs(rng.normal(loc=0, scale=1, size=n_anchor_rows)) for _ in range(n_anchor_cols) ]
		U = np.array([ U[row_iter]/np.linalg.norm(U[row_iter], ord=1) for row_iter in range(n_anchor_cols) ])
		
		################### Plot entire M_test Matrix
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_anchor_rows,n_anchor_cols))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(U, cmap=U_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_U.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
		
		
		
		
		
		
		
		itemEmbed = [ np.abs(rng.normal(loc=0, scale=1, size=n_cols)) for _ in range(n_anchor_cols) ]
		itemEmbed = np.array([ itemEmbed[row_iter]/np.linalg.norm(itemEmbed[row_iter], ord=1) for row_iter in range(n_anchor_cols) ])
		
		################### Plot entire ItemEmbed Matrix
		plt.clf()
		fig, ax = plt.subplots(figsize=(n_cols,n_anchor_cols))
		
		# norm = plt.Normalize(vmin=0, vmax=10, clip=True)
		norm = plt.Normalize(vmin=0, vmax=vmax, clip=True)
		im = ax.imshow(itemEmbed, cmap=itemEmbed_cmap, norm=norm)
		ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

		fig.tight_layout()
		out_fname = f"{res_dir}/1_itemEmbed.{extension}"
		Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
		plt.savefig(f"{out_fname}",bbox_inches='tight')
		plt.close()
		os.system(f"pdfcrop {out_fname} {out_fname}")
		
	except Exception as e:
		embed()
		raise e

	
	

def main():
	
	base_res_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/models"
	data_dir = "../../data/zeshel"
	
	parser = argparse.ArgumentParser(description='Make plots for CUR paper')
	parser.add_argument("--rq", type=int, required=True, choices=[-1,0,1,2,3,4,5,6,7,8], help="Mode values controls type of plot made")
	
	args = parser.parse_args()
	rq = args.rq
	
	LOGGER.info(f"Plotting results for RQ = {rq}")
	
	if rq == -1:
		plot_heatmap_for_cur_diagram(res_dir=f"{base_res_dir}/plots/CUR_Diagram")
	elif rq == 1:

		# nm_train_vals  = [500]
		domain_vals  = ["yugioh", "doctor_who", "pro_wrestling", "military", "star_trek"]
		nm_train_vals  = [100, 500, 2000]
		# domain_vals  = ["yugioh"]
		
		topk_vals = [1, 10, 50, 100]
		for nm_train, domain in itertools.product(nm_train_vals, domain_vals):
			if domain == "pro_wrestling":
				nm_train = min(1000, nm_train)
			try:
				plot_rq_1_2_performance_vs_topk_retrieved_or_cost(
					base_res_dir=base_res_dir,
					nm_train=nm_train,
					domain=domain,
					topk_vals=topk_vals,
					rq_name="RQ1_Model_Performance_At_Equal_Num_Retrieved"
				)
			except Exception as e:
				LOGGER.info(f"Error for nm_train = {nm_train}, domain = {domain}")
				
	elif rq == 2:

		nm_train_vals  = [100, 500, 2000]
		domain_vals  = ["yugioh", "doctor_who", "pro_wrestling", "military", "star_trek"]
		# nm_train_vals  = [500]
		# domain_vals  = ["yugioh"]
		
		topk_vals = [1, 10, 50, 100]
		# topk_vals = [100]
		for nm_train, domain in itertools.product(nm_train_vals, domain_vals):
			if domain == "pro_wrestling":
				nm_train = min(1000, nm_train)
			try:
				plot_rq_1_2_performance_vs_topk_retrieved_or_cost(
					base_res_dir=base_res_dir,
					nm_train=nm_train,
					domain=domain,
					topk_vals=topk_vals,
					rq_name="RQ2_Model_Performance_At_Equal_Test_Cost"
				)
			except Exception as e:
				LOGGER.info(f"Error for nm_train = {nm_train}, domain = {domain}")
	
	elif rq == 5:

		# nm_train_vals  = [100, 500, 2000]
		# domain_vals  = ["yugioh", "doctor_who", "pro_wrestling", "military", "star_trek"]
		nm_train_vals  = [500]
		domain_vals  = ["yugioh"]
		
		topk_vals = [1, 10, 50, 100]
		# topk_vals = [100]
		for nm_train, domain in itertools.product(nm_train_vals, domain_vals):
			try:
				plot_rq_5_6_performance_vs_topk_retrieved_or_cost_for_CE_only_baselines(
					base_res_dir=base_res_dir,
					nm_train=nm_train,
					domain=domain,
					topk_vals=topk_vals,
					rq_name="RQ5_Model_Performance_At_Equal_Test_Cost_CE_Baselines"
				)
			except Exception as e:
				LOGGER.info(f"Error for nm_train = {nm_train}, domain = {domain}")
	
	elif rq == 6:

		# nm_train_vals  = [100, 500, 2000]
		# domain_vals  = ["yugioh", "doctor_who", "pro_wrestling", "military", "star_trek"]
		nm_train_vals  = [500]
		domain_vals  = ["yugioh"]
		
		topk_vals = [1, 10, 50, 100]
		# topk_vals = [100]
		for nm_train, domain in itertools.product(nm_train_vals, domain_vals):
			try:
				plot_rq_5_6_performance_vs_topk_retrieved_or_cost_for_CE_only_baselines(
					base_res_dir=base_res_dir,
					nm_train=nm_train,
					domain=domain,
					topk_vals=topk_vals,
					rq_name="RQ6_Model_Performance_At_Equal_Num_Retrieved_CE_Baselines"
				)
			except Exception as e:
				LOGGER.info(f"Error {e} for nm_train = {nm_train}, domain = {domain}")
				
	elif rq == 3:
		topk_vals  = [1, 100]
		cost_vals  = [10, 500]
		nm_train_vals  = [100, 500, 2000]
		nm_train_vals  = [100, 500]
		
		topk_vals  = [100]
		cost_vals  = [500]
		nm_train_vals  = [500]
		
		for topk, cost, nm_train in itertools.product(topk_vals, cost_vals, nm_train_vals):
			if topk > cost: continue
			plot_rq_3_performance_vs_domains_size(
				res_dir=base_res_dir,
				topk=topk,
				cost=cost,
				nm_train=nm_train
			)
	
	elif rq == 4:
		topk_vals  = [1, 100]
		cost_vals  = [100, 500, 1000]
		
		topk_vals  = [100]
		cost_vals  = [100, 500]
		
		eval_data_vals = ["train", "test"]
		
		domain_vals = {
			"pro_wrestling":"Pro-Wrestling",
			"yugioh":"YuGiOh",
			"star_trek":"Star-Trek",
			"doctor_who":"Doctor-Who",
			"military":"Military"
		}
		
		eval_data_vals = ["test"]
		for topk, cost, eval_data in itertools.product(topk_vals, cost_vals, eval_data_vals):
			if topk >= cost: continue
			# if cost == 2000 and domain ==  "pro_wrestling": continue
			# if cost == 1000 and domain !=  "pro_wrestling": continue
			
			plot_rq_4_performance_vs_train_data_size(
				res_dir=base_res_dir,
				topk=topk,
				cost=cost,
				domain_vals=domain_vals,
				eval_data=eval_data
			)
	
	elif rq == 7:
		
		eval_methods  = ["cur", "cur_oracle"]
		data_file_e_crossenc_6_256 = "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt/yugioh/Retrieval_wrt_Exact_CrossEnc/nm=3374_ne=10031_s=5_for_plots/retrieval_wrt_exact_crossenc.json"
		data_file_cls_crossenc_6_49_rep = "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt/yugioh/Retrieval_wrt_Exact_CrossEnc/nm=3374_ne=10031_s=5_for_plots/retrieval_wrt_exact_crossenc.json"
		
		# data_file_e_crossenc_6_256 = "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt/yugioh/Retrieval_wrt_Exact_CrossEnc/nm=3374_ne=10031_s=5_for_approx_error/retrieval_wrt_exact_crossenc.json"
		# data_file_cls_crossenc_6_49_rep = "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/score_mats_model-1-12279.0--80.14.ckpt/yugioh/Retrieval_wrt_Exact_CrossEnc/nm=3374_ne=10031_s=5_for_approx_error/retrieval_wrt_exact_crossenc.json"
	
	
		out_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/models/plots/RQ7_CLS_vs_E_CrossEnc"
		for ce_type, data_file in [("ece", data_file_e_crossenc_6_256), ("cls_ce", data_file_cls_crossenc_6_49_rep)]:
			plot_rq7_heatmaps(
				data_file=data_file,
				method_vals=eval_methods,
				out_dir=f"{out_dir}/{ce_type}",
				crossenc_type=ce_type
			)
	
	elif rq == 0:
		
		out_dir = "../../results/8_CUR_EMNLP/d=ent_link_ce/models/plots/RQ0_CLS_vs_E_CrossEnc_Score_Distribution"
		
		# for bins in [100, 200, 500, 1000]:
		for bins in [200]:
			plot_rq_0_score_distribution(
				out_dir=out_dir,
				bins=bins,
				plot_per_row=False
			)
	
	else:
		raise NotImplementedError(f"RQ = {rq} not supported")


if __name__ == "__main__":
	main()

