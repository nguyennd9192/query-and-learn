from params import *
from utils.plot import *
from general_lib import *

import sys, pickle, functools, json, copy
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
# performance_codes = dict({"uniform":"red", "exploitation":"black", 
# 		"margin":"blue", "expected_improvement":"green"})

performance_codes = dict({"org_space":"red", "MLKR":"blue"})

def perform_each_acquisition(ith_trials, embedding_method, sampling_method, dt, ax):
	# # loop all queries in each state
	mean_vals = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
	mean_pos = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})

	FLAGS.embedding_method = embedding_method
	FLAGS.sampling_method = sampling_method

	qids = range(1, n_run)
	job_savedir = get_savedir(ith_trial=ith_trials[0])

	for qid in qids:
		values = []
		for ith_trial in ith_trials:
			savedir = job_savedir.replace("/trial_1/", "/trial_{}/".format(ith_trial))
			eval_file = savedir + "/query_{0}/eval_query_{0}.pkl".format(qid)

			if not gfile.exists(eval_file):
				continue
			# else:
			# 	print ("File existed.")

			data = load_pickle(eval_file)
			dict_values = data[dt]
			idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]
			if dt == "DQ_to_RND":
				is_shown_tails = False
			else:
				is_shown_tails = True
			# ax, y_star_ax, mean, y_min = show_one_rst(y=y_qr, y_pred=y_qr_pred, 
			# 	ax=ax, y_star_ax=y_star_ax, ninst_ax=ninst_ax,
			# 	pos_x=pos_x, color=performance_codes[method],	is_shown_tails=is_shown_tails)
			error = np.abs(y_qr - y_qr_pred)
			mean = np.mean(error)
			if embedding_method == "org_space":
				print (embedding_method, mean)
			y_min = np.min(y_qr)
			if dt == "OS":
				values.append(y_min)
			else:
				values.append(mean)

		bplot = ax.boxplot(x=values, vert=True, #notch=True, 
				# sym='rs', # whiskerprops={'linewidth':2},
				positions=[qid], patch_artist=True,
				widths=0.25, meanline=True, #flierprops=flierprops,
				showfliers=False, showbox=True, showmeans=False)
			# ax.text(pos_x, mean, round(mean, 2),
			# 	horizontalalignment='center', size=14, 
			# 	color=color, weight='semibold')
		patch = bplot['boxes'][0]
		patch.set_facecolor(performance_codes[embedding_method])

			# mean_vals[dt].append(mean)
			# mean_pos[dt].append(qid)
	return mean_vals, mean_pos



def show_performance(ith_trials, dt):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.grid(which='both', linestyle='-.')
	ax.grid(which='minor', alpha=0.2)


	qids = range(1, n_run)
	mean_perform = dict({})
	for embedding_method in ["org_space", "MLKR"]: # 
		for sampling_method in ["uniform", "exploitation", "margin"]: # expected_improvement
			flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
						  linestyle='none', markeredgecolor='k')
			mean_vals, mean_pos = perform_each_acquisition(ith_trials=ith_trials,
				embedding_method=embedding_method,
				sampling_method=sampling_method, dt=dt, ax=ax)

		# ax.set_xlabel(r"Query index", **axis_font) 
		if dt == "OS":
			ax.set_ylabel(r"min(y_qeried)", **axis_font)
			ax.set_ylim(-1.6, -0.4)

		else:
			ax.set_ylabel(r"|y_obs - y_pred|", **axis_font)
			ax.set_yscale('log')
			# ax.set_ylim(0.001, 1.3)
	
	plt.xticks(qids, qids) 
	plt.legend()
	ax.tick_params(axis='y', labelsize=12)
	# y_star_ax.set_yscale('log')
	ax.set_title(dt)

	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/box/"+dt+".pdf"
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)


	# # plot relative
	# fig2 = plt.figure(figsize=(10, 8))
	# ax2 = fig2.add_subplot(1, 1, 1)
	# ax2.grid(which='both', linestyle='-.')
	# ax2.grid(which='minor', alpha=0.2)
	# y_uniform = mean_perform["uniform"]["y"]
	# for method in ["uniform", "exploitation", "margin"]: # expected_improvement
	# 	x = mean_perform[method]["x"]
	# 	y = mean_perform[method]["y"]

	# 	ax2.plot(x, np.divide(y, y_uniform), color=performance_codes[method], 
	# 		alpha=0.8, linestyle="-.", marker="o",
	# 		label=method)
	# ax2.set_yscale('log')
	# plt.xticks(qids, qids)
	# plt.legend()
	# ax2.tick_params(axis='y', labelsize=12)
	# ax2.set_title(dt)
	# plt.tight_layout(pad=1.1)
	# save_at = result_dropbox_dir+"/merge_performance/"+"/{}/".format(FLAGS.score_method)+dt+"_ratio.pdf"
	# makedirs(save_at)
	# plt.savefig(save_at, transparent=False)
	# print ("save_at: ", save_at)
if __name__ == "__main__":
	FLAGS(sys.argv)
	# pr_file = sys.argv[-1]
	# kwargs = load_pickle(filename=pr_file)
	# FLAGS.score_method = kwargs["score_method"]
	# FLAGS.sampling_method =	kwargs["sampling_method"]
	# FLAGS.embedding_method = kwargs["embedding_method"]
	# FLAGS.active_p = kwargs["active_p"]
	# FLAGS.ith_trial = kwargs["ith_trial"]

	for dt in ["DQ", "OS", "RND", "DQ_to_RND"]:
		show_performance(ith_trials=[1,2,3], dt=dt)

		