from params import *
from utils.plot import *

import sys, pickle, functools, json
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np

axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}


def show_performance(ith_trial):
	unlbl_job = "mix" # mix, "mix_2-24"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 
	unlbl_dir = result_file.replace(".pkl","")+"/"+unlbl_job

	qids = range(1, 30)
	# qids = [1]
	eval_files = [unlbl_dir+"/query_{0}/eval_query_{0}.pkl".format(qid) for qid in qids]
	fig = plt.figure(figsize=(10, 8))
	grid = plt.GridSpec(6, 4, hspace=0.3, wspace=0.3)
	ax = fig.add_subplot(grid[:3, :], xticklabels=[])
	y_star_ax = fig.add_subplot(grid[3:5, :], sharex=ax)
	ninst_ax = fig.add_subplot(grid[-1:, :], sharex=ax)

	flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
				  linestyle='none', markeredgecolor='k')
	
	dx = 0.2
	# # loop all queries in each state
	mean_vals = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
	mean_pos = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
	for qid, eval_file in zip(qids, eval_files):
		# eval_file = '/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/mix/query_1/eval_query_1.pkl'
		data = load_pickle(eval_file)
		for dt, dict_values in data.items():
			idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]
			pos_x = qid + pos_codes[dt]*dx
			if dt == "DQ_to_RND":
				is_shown_tails = False
			else:
				is_shown_tails = True
			ax, y_star_ax, mean, y_min = show_one_rst(y=y_qr, y_pred=y_qr_pred, 
				ax=ax, y_star_ax=y_star_ax, ninst_ax=ninst_ax,
				pos_x=pos_x, color=color_codes[dt],	is_shown_tails=is_shown_tails)
			mean_vals[dt].append(mean)
			mean_pos[dt].append(pos_x)

			if dt == "OS":
				print ("qid: ", qid, len(idx_qr), mean, y_min)
		# print (dir(data))
		# break

	for dt, dict_values in mean_vals.items():
		x = mean_pos[dt]
		y = mean_vals[dt]
		ax.plot(x, y, color=color_codes[dt], alpha=0.8, linestyle="-.")

	ax.set_yscale('log')
	# ax.set_xlabel(r"Query index", **axis_font) 
	ax.set_ylabel(r"|y_obs - y_pred|", **axis_font)
	y_star_ax.set_ylabel(r"min(y_os)", **axis_font)
	ninst_ax.set_ylabel(r"n_qr", **axis_font)
	plt.xticks(qids, qids)

	ax.tick_params(axis='y', labelsize=12)
	# y_star_ax.set_yscale('log')

	plt.setp(ax.get_xticklabels(), visible=False)
	plt.setp(y_star_ax.get_xticklabels(), visible=False)

	plt.tight_layout(pad=1.1)
	save_at = unlbl_dir+"/all_query_performance.pdf"

	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)
if __name__ == "__main__":
	FLAGS(sys.argv)
	show_performance(ith_trial="000")
