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

performance_codes = dict({"org_space":"blue", "MLKR":"red"}) 
hatch_codes = dict({"uniform":"/", "exploitation":"*", 
				"margin":"o", "expected_improvement":"/", "MaxEmbeddDir":"."}) 
# # '-', '+', 'x', '\\', '*', 'o', 'O', '.'

def get_error(job_savedir, ith_trial, qid):
	savedir = job_savedir.replace("/trial_1/", "/trial_{}/".format(ith_trial))
		
	# # read load only rand query
	eval_file = savedir + "/query_{0}/eval_query_{0}.pkl".format(qid)
	data = load_pickle(eval_file)

	if dt == "DU":
		dict_values = data["DQ"]
	else:
		dict_values = data[dt]
	idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]
	if dt == "DQ_to_RND":
		is_shown_tails = False
	else:
		is_shown_tails = True
	# ax, y_star_ax, mean, y_min = show_one_rst(y=y_qr, y_pred=y_qr_pred, 
	# 	ax=ax, y_star_ax=y_star_ax, ninst_ax=ninst_ax,
	# 	pos_x=pos_x, color=performance_codes[method],	is_shown_tails=is_shown_tails)
	error_qr = np.abs(y_qr - y_qr_pred)
	y_min = np.min(y_qr)

	# # read load the rest
	query_file = savedir + "/query_{0}/query_{0}.csv".format(qid)
	# if not gfile.exists(query_file):
	# 	continue
	df = pd.read_csv(query_file, index_col=0)
	ids_non_qr = df[df["last_query_{}".format(qid)].isnull()].index.tolist()
	error_non_qr = df.loc[ids_non_qr, "err_{}".format(qid)]
	
	mean_qr = np.mean(error_qr)
	mean_non_qr = np.mean(error_non_qr)

	return mean_qr, mean_non_qr, data

def perform_each_acquisition(ith_trials, 
		embedding_method, sampling_method, dt, ax, is_relative):
	# # loop all queries in each state
	mean_vals = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})
	mean_pos = dict({"DQ":[], "OS":[], "RND":[], "DQ_to_RND":[]})

	if is_relative:
		FLAGS.embedding_method = "org_space"
		FLAGS.sampling_method = "uniform"

		org_savedir = get_savedir(ith_trial=ith_trials[0])


	FLAGS.embedding_method = embedding_method
	FLAGS.sampling_method = sampling_method

	qids = range(1, n_run)

	if dt == "OS":
		full_os_ids = get_full_os()
		n_full = len(full_os_ids)

	recall_os = 0
	n_trials = len(ith_trials)
	job_savedir = get_savedir(ith_trial=ith_trials[0])

	for qid in qids:
		values = []
		for ith_trial in ith_trials:
			mean_qr, mean_non_qr, data = get_error(
				job_savedir=job_savedir, ith_trial=ith_trial, qid=qid)
			if is_relative:
				mean_qr_ref, mean_non_qr_ref, data_ref = get_error(
					job_savedir=org_savedir, ith_trial=ith_trial, qid=qid)
				mean_qr /= mean_qr_ref
				mean_non_qr /= mean_non_qr_ref

			if dt == "OS":
				for tmp in ["DQ", "OS", "RND"]:
					tmp_dict_values = data[tmp]
					idx_expl, y_expl, y_expl_pred = tmp_dict_values["idx_qr"], tmp_dict_values["y_qr"], tmp_dict_values["y_qr_pred"]
					this_recall = len(list(set(idx_expl).intersection(full_os_ids))) / float(n_full * n_trials)
					recall_os += this_recall
				values.append(recall_os)
			elif dt == "DU":
				values.append(mean_non_qr)
			else:
				values.append(mean_qr)


		bplot = ax.boxplot(x=values, vert=True, #notch=True, 
				# sym='rs', # whiskerprops={'linewidth':2},
				# alpha=0.4,
				# notch=True,
				positions=[qid], patch_artist=True,
				widths=0.8, meanline=True, #flierprops=flierprops,
				showfliers=False, showbox=True, showmeans=False,
				)
			# ax.text(pos_x, mean, round(mean, 2),
			# 	horizontalalignment='center', size=14, 
			# 	color=color, weight='semibold')
		patch = bplot['boxes'][0]
		patch.set_facecolor(performance_codes[embedding_method])
		patch.set_hatch(hatch_codes[sampling_method])
		patch.set_alpha(0.8)
		# ax.legend([bplot["boxes"][0]], 
		# 	["{0}_{1}".format(embedding_method, sampling_method)], 
		# 	loc='upper right')



			# mean_vals[dt].append(mean)
			# mean_pos[dt].append(qid)
	return mean_vals, mean_pos, patch



def show_performance(ith_trials, dt, is_relative):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.grid(which='both', linestyle='-.')
	ax.grid(which='minor', alpha=0.2)


	qids = range(1, n_run)
	mean_perform = dict({})

	legends = []
	patches = []
	for embedding_method in ["org_space", "MLKR"]: # 
		for sampling_method in ["margin", "exploitation", "uniform"]: # , "exploitation", "margin"
		# expected_improvement,  MaxEmbeddDir
			flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
						  linestyle='none', markeredgecolor='k')
			try:
				mean_vals, mean_pos, patch = perform_each_acquisition(
					ith_trials=ith_trials,
					embedding_method=embedding_method,
					sampling_method=sampling_method, dt=dt, ax=ax,
					is_relative=is_relative)
				lab = "{0}|{1}".format(embedding_method, sampling_method)
				legends.append(lab)
				patches.append(patch)
			except:
				pass

	

		# ax.set_xlabel(r"Query index", **axis_font) 
		if dt == "OS":
			ax.set_ylabel(r"Recall rate", **axis_font)
			# ax.set_ylim(-1.6, -0.4)

		else:
			ax.set_ylabel(r"Error", **axis_font)
			ax.set_yscale('log')
			# ax.set_ylim(0.001, 1.3)
	ax.legend(patches, legends)
	
	plt.xticks(qids, qids) 
	ax.tick_params(axis='y', labelsize=12)
	# y_star_ax.set_yscale('log') 
	ax.set_title(dt)

	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/box/"+dt+".pdf"
	if is_relative:
		save_at = result_dropbox_dir+"/merge_performance/box_rev/"+dt+".pdf"
	
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




def get_full_os():
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = load_data()
	lim_os = -0.05
	ids = np.where(unlbl_y < lim_os)[0]

	print (unlbl_y[ids])
	print (len(unlbl_y[ids]))
	# print (unlbl_index[ids])
	# x = np.concatenate((X_train[:, 28], unlbl_X[:, 28]), axis=0)
	# y = np.concatenate((y_train, unlbl_y), axis=0)

	# print (x)
	# print (y)


	# scatter_plot(x=x, y=y, xvline=None, yhline=None, 
	# 	sigma=None, mode='scatter', lbl=None, name=None, 
	# 	x_label='x', y_label='y', 
	# 	save_file=result_dropbox_dir+"/test.pdf", interpolate=False, color='blue', 
	# 	linestyle='-.', 
	# 	marker=['o']*len(y), title=None)
	return unlbl_index[ids]



if __name__ == "__main__":
	FLAGS(sys.argv)
	# pr_file = sys.argv[-1]
	# kwargs = load_pickle(filename=pr_file)
	# FLAGS.score_method = kwargs["score_method"]
	# FLAGS.sampling_method =	kwargs["sampling_method"]
	# FLAGS.embedding_method = kwargs["embedding_method"]
	# FLAGS.active_p = kwargs["active_p"]
	# FLAGS.ith_trial = kwargs["ith_trial"]

	# get_full_os()

	for dt in ["DQ", "OS", "RND", "DQ_to_RND", "DU"]: # "DQ", "OS", "RND", "DQ_to_RND", "DU"
		show_performance(ith_trials=[1,2,3,4,5], # 2,3,4,5,
			# 2
			dt=dt, is_relative=False)

		