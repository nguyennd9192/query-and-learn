from params import *
from utils.plot import *

import sys, pickle, functools, json, copy
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}


def show_performance(ith_trial, dt):
	unlbl_job = "mix" # mix, "mix_2-24"

	performance_codes = dict({"uniform":"red", "exploitation":"black", 
					"margin":"blue", "expected_improvement":"green"})
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.grid(which='both', linestyle='-.')
	ax.grid(which='minor', alpha=0.2)
	
	qids = range(1, 26)
	mean_perform = dict({})
	for method in ["uniform", "exploitation", "margin"]: # expected_improvement
		FLAGS.sampling_method =method
		result_dir = get_savedir()
		filename = get_savefile()

		result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
		unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 
		unlbl_dir = result_file.replace(".pkl","")+"/"+unlbl_job

		# qids = [1]
		eval_files = [unlbl_dir+"/query_{0}/eval_query_{0}.pkl".format(qid) for qid in qids]


		flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
					  linestyle='none', markeredgecolor='k')
		
		dx = 0.2
		# # loop all queries in each state
		mean_vals = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
		mean_pos = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
		for qid, eval_file in zip(qids, eval_files):
			data = load_pickle(eval_file)
			# for dt, dict_values in data.items():
			dict_values = data[dt]
			idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]
			pos_x = qid #+ performance_pos[method]*dx
			if dt == "DQ_to_RND":
				is_shown_tails = False
			else:
				is_shown_tails = True
			# ax, y_star_ax, mean, y_min = show_one_rst(y=y_qr, y_pred=y_qr_pred, 
			# 	ax=ax, y_star_ax=y_star_ax, ninst_ax=ninst_ax,
			# 	pos_x=pos_x, color=performance_codes[method],	is_shown_tails=is_shown_tails)
			error = np.abs(y_qr - y_qr_pred)
			mean = np.mean(error)
			y_min = np.min(y_qr)

			if dt == "OS":
				x = y_qr
				mean = np.mean(y_qr)
			else:
				x = error
			# bplot = ax.boxplot(x=x, vert=True, #notch=True, 
			# 		# sym='rs', # whiskerprops={'linewidth':2},
			# 		positions=[pos_x], patch_artist=True,
			# 		widths=0.1, meanline=True, #flierprops=flierprops,
			# 		showfliers=False, showbox=True, showmeans=False)
			# 	# ax.text(pos_x, mean, round(mean, 2),
			# 	# 	horizontalalignment='center', size=14, 
			# 	# 	color=color, weight='semibold')
			# patch = bplot['boxes'][0]
			# patch.set_facecolor(performance_codes[method])

			mean_vals[dt].append(mean)
			mean_pos[dt].append(pos_x)

			# break
		x = mean_pos[dt]
		y = mean_vals[dt]
		mean_perform[method] = dict({})
		mean_perform[method]["x"] = x
		mean_perform[method]["y"] = y

		ax.plot(x, y, color=performance_codes[method], 
			alpha=0.8, linestyle="-.", marker="o",
			label=method)


	# ax.set_xlabel(r"Query index", **axis_font) 
	if dt == "OS":
		ax.set_ylabel(r"min(y_qeried)", **axis_font)
		ax.set_ylim(-1.6, -0.4)

	else:
		ax.set_ylabel(r"|y_obs - y_pred|", **axis_font)
		ax.set_yscale('log')
		ax.set_ylim(0.001, 1.3)
	plt.xticks(qids, qids)
	plt.legend()
	ax.tick_params(axis='y', labelsize=12)
	# y_star_ax.set_yscale('log')
	ax.set_title(dt)

	# plt.setp(ax.get_xticklabels(), visible=False)

	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/"+"/{}/".format(FLAGS.score_method)+dt+".pdf"
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)


	# # plot relative
	fig2 = plt.figure(figsize=(10, 8))
	ax2 = fig2.add_subplot(1, 1, 1)
	ax2.grid(which='both', linestyle='-.')
	ax2.grid(which='minor', alpha=0.2)
	y_uniform = mean_perform["uniform"]["y"]
	for method in ["uniform", "exploitation", "margin"]: # expected_improvement
		x = mean_perform[method]["x"]
		y = mean_perform[method]["y"]

		ax2.plot(x, np.divide(y, y_uniform), color=performance_codes[method], 
			alpha=0.8, linestyle="-.", marker="o",
			label=method)
	ax2.set_yscale('log')
	plt.xticks(qids, qids)
	plt.legend()
	ax2.tick_params(axis='y', labelsize=12)
	ax2.set_title(dt)
	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/"+"/{}/".format(FLAGS.score_method)+dt+"_ratio.pdf"
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)
if __name__ == "__main__":
	FLAGS(sys.argv)
	for dt in ["DQ", "OS", "RND", "DQ_to_RND"]:
		show_performance(ith_trial="000", dt=dt)
