from params import *
from utils.plot import *
from general_lib import *

import sys, pickle, functools, json, copy
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}

performance_codes = dict({
	"org_space|uniform":"teal", "org_space|margin":"darkslategrey", "org_space|exploitation":"paleturquoise", 
	"MLKR|uniform":"maroon", "MLKR|margin":"indianred", "MLKR|exploitation":"tomato", 
	}) 

hatch_codes = dict({"uniform":"/", "exploitation":"*", 
				"margin":"o", "expected_improvement":"/", "MaxEmbeddDir":"."}) 

def get_error(job_savedir, ith_trial, qid, dt):
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

	return error_qr, error_non_qr, data

def perform_each_acquisition(ith_trials, 
		embedding_method, sampling_method, dt, ax, is_relative, ax2=None):
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
	job_savedir = get_savedir(ith_trial=ith_trials[0])

	mean_trials = []
	var_trials = []
	patch = None
	for qid in qids:
		values = []
		# # for all trials
		n_trials = 0
		for ith_trial in ith_trials:
			try:
				mean_qr, mean_non_qr, data = get_error(
					job_savedir=job_savedir, ith_trial=ith_trial, qid=qid, dt=dt)
				if is_relative:
					mean_qr_ref, mean_non_qr_ref, data_ref = get_error(
						job_savedir=org_savedir, ith_trial=ith_trial, qid=qid, dt=dt)
					mean_qr /= mean_qr_ref
					mean_non_qr /= mean_non_qr_ref

				if dt == "OS":
					for tmp in ["DQ", "OS", "RND"]:
						tmp_dict_values = data[tmp]
						idx_expl, y_expl, y_expl_pred = tmp_dict_values["idx_qr"], tmp_dict_values["y_qr"], tmp_dict_values["y_qr_pred"]
						this_recall = len(list(set(idx_expl).intersection(full_os_ids))) / float(n_full)
						recall_os += this_recall
					values.append(recall_os)
				elif dt == "DU":
					values.append(mean_non_qr)
				else:
					values.append(mean_qr)
				n_trials += 1

			except Exception as e:
				print ("Error in ", embedding_method, sampling_method, ith_trial, qid)
				pass
		values = np.array(values)
		if dt == "OS":
			# # normalize in scale of outstanding 
			values /= n_trials


		mean_trial = np.mean(values)
		var_trial = np.var(values)
		print (values, mean_trial, var_trial)


		if dt == "OS": # OS
			lab = "{0}|{1}".format(embedding_method, sampling_method)
			print ("values.shape: ", values.shape)

			bplot1 = ax.boxplot(x=np.ravel(values), vert=True, #notch=True, 
					# sym='rs', # whiskerprops={'linewidth':2},
					# alpha=0.4,
					# notch=True,
					positions=[qid], patch_artist=True, notch=True, 
					widths=0.8, meanline=False, #flierprops=flierprops,
					showfliers=False, showbox=True, manage_ticks=False,
					showmeans=False, showcaps=False
					)
			bplot2 = ax2.boxplot(x=np.ravel(values), vert=True, #notch=True, 
					# sym='rs', # whiskerprops={'linewidth':2},
					# alpha=0.4,
					# notch=True,
					positions=[qid], patch_artist=True, notch=True, 
					widths=0.8, meanline=False, #flierprops=flierprops,
					showfliers=False, showbox=True, manage_ticks=False,
					showmeans=False, showcaps=False
					)
			for bplot in (bplot1, bplot2):
				for patch in bplot['boxes']:
					patch.set_facecolor(performance_codes[lab])
					patch.set(color=performance_codes[lab], linewidth=0.1, alpha=0.1)
					# patch.set(color=None)
					patch.set_alpha(0.6)


		mean_trials.append(mean_trial)
		var_trials.append(var_trial)

	return np.array(mean_trials), np.array(var_trials), patch



def show_performance(ith_trials, dt, is_relative):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	fig = plt.figure(figsize=(11, 8))
	ax = fig.add_subplot(1, 1, 1)

	# plot the same data on both axes
	ax1.plot(pts)
	ax2.plot(pts)


	qids = range(1, n_run)
	mean_perform = dict({})

	legends = []
	patches = []

	linestyle_dict = dict({"margin":":", "exploitation":"-.", "uniform":"-"})
	for embedding_method in ["MLKR", "org_space"]: # "org_space"
		for sampling_method in ["margin", "exploitation", "uniform"]: # ,  
		# expected_improvement,  MaxEmbeddDir
			flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
						  linestyle='none', markeredgecolor='k')
			# try:
			mean_trials, var_trials, patch = perform_each_acquisition(
				ith_trials=ith_trials,
				embedding_method=embedding_method,
				sampling_method=sampling_method, dt=dt, ax=ax,
				is_relative=is_relative)
			lab = "{0}|{1}".format(embedding_method, sampling_method)
			legends.append(lab)
			patches.append(patch)
			if dt != "OS":
				ax.plot(qids, mean_trials, '-', c=performance_codes[lab], 
					label=lab, linestyle=linestyle_dict[sampling_method], linewidth=2)
				ax.fill_between(qids, mean_trials - var_trials, mean_trials + var_trials, 
					color=performance_codes[lab],
					alpha=0.8)


		if dt == "OS":
			ax.set_ylabel(r"Recall rate", **axis_font)
		
	ax.legend()
	
	plt.xticks(qids, qids) 
	ax.tick_params(axis='y', labelsize=12)
	ax.set_title(dt)


	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/box/"+dt+".pdf"
	if is_relative:
		save_at = result_dropbox_dir+"/merge_performance/box_rev/"+dt+".pdf"
	
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)


def show_performance_OS(ith_trials, dt, is_relative):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
	fig.subplots_adjust(hspace=0.02)  # adjust space between axes


	qids = range(1, n_run)
	mean_perform = dict({})

	legends = []
	patches = []

	linestyle_dict = dict({"margin":":", "exploitation":"-.", "uniform":"-"})
	for embedding_method in ["MLKR", "org_space"]: # "org_space"
		for sampling_method in ["margin", "exploitation", "uniform"]: # ,  
			flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
						  linestyle='none', markeredgecolor='k')
			mean_trials, var_trials, patch = perform_each_acquisition(
				ith_trials=ith_trials,
				embedding_method=embedding_method,
				sampling_method=sampling_method, dt=dt, ax=ax1, ax2=ax2,
				is_relative=is_relative)
			lab = "{0}|{1}".format(embedding_method, sampling_method)
			legends.append(lab)
			patches.append(patch)

	# zoom-in / limit the view to different portions of the data
	ax1.set_ylim(0.8, 1.0)  # most of the data
	ax2.set_ylim(0.0, 0.5)  # outliers only 

	# hide the spines between ax and ax2
	ax1.spines['bottom'].set_visible(False) 
	ax2.spines['top'].set_visible(False)

	# ax1.xaxis.tick_bottom()
	ax1.xaxis.tick_top()
	ax2.tick_params(labeltop=False)  # don't put tick labels at the top

	d = .5  # proportion of vertical to horizontal extent of the slanted line
	kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
				  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
	ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
	ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

	ax1.set_yticks([0.8, 0.9, 1.0])
	ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])

	# plt.xticks(qids, qids) 

	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/box_DQ_OS_RND/"+dt+".pdf"
	
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)


def show_performance_DU(ith_trials, dt, is_relative):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2]})
	fig.subplots_adjust(hspace=0.02)  # adjust space between axes


	qids = range(1, n_run)
	mean_perform = dict({})

	legends = []
	patches = []

	linestyle_dict = dict({"margin":":", "exploitation":"-.", "uniform":"-"})
	for embedding_method in ["MLKR", "org_space"]: # "org_space"
		for sampling_method in ["margin", "exploitation", "uniform"]: # ,  
			flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
						  linestyle='none', markeredgecolor='k')
			mean_trials, var_trials, patch = perform_each_acquisition(
				ith_trials=ith_trials,
				embedding_method=embedding_method,
				sampling_method=sampling_method, dt=dt, ax=ax1,
				is_relative=is_relative)
			lab = "{0}|{1}".format(embedding_method, sampling_method)
			legends.append(lab)
			patches.append(patch)

			# # plot the same data on both axes
			ax1.plot(qids, mean_trials, c=performance_codes[lab], 
				 linestyle="-.", linewidth=1)
			ax1.fill_between(qids, mean_trials - 2*var_trials, mean_trials + 2*var_trials, 
				color=performance_codes[lab], label=lab,
				alpha=0.6)

			ax2.plot(qids, mean_trials, c=performance_codes[lab], 
				label=lab, linestyle="-.", linewidth=1)
			ax2.fill_between(qids, mean_trials - 2*var_trials, mean_trials + 2*var_trials, 
				color=performance_codes[lab],
				alpha=0.6)


	ax1.tick_params(axis='y', labelsize=12)
	# ax1.set_title(dt)

	# zoom-in / limit the view to different portions of the data
	ax1.set_ylim(0.045, 0.075)  # most of the data
	ax2.set_ylim(0.01, 0.032)  # outliers only 


	# hide the spines between ax and ax2
	ax1.spines['bottom'].set_visible(False) 
	ax2.spines['top'].set_visible(False)

	# ax1.xaxis.tick_bottom()
	ax1.xaxis.tick_top()
	ax2.tick_params(labeltop=False)  # don't put tick labels at the top

	d = .5  # proportion of vertical to horizontal extent of the slanted line
	kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
				  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
	ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
	ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

	# ax1.set_yticks([])
	# ax2.set_yticks([])
	# ax2.set_xticks([])
	ax2.set_yticks([0.01, 0.02, 0.03])
	ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])
	
	plt.tight_layout(pad=1.1)
	save_at = result_dropbox_dir+"/merge_performance/box_DQ_OS_RND/"+dt+".pdf"
	if is_relative:
		save_at = result_dropbox_dir+"/merge_performance/box_rev/"+dt+".pdf"
	
	makedirs(save_at)
	plt.savefig(save_at, transparent=False)
	print ("save_at: ", save_at)


def get_full_os():
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = load_data()
	lim_os = -0.1 # -0.05
	ids = np.where(unlbl_y < lim_os)[0]

	# scatter_plot(x=x, y=y, xvline=None, yhline=None, 
	# 	sigma=None, mode='scatter', lbl=None, name=None, 
	# 	x_label='x', y_label='y', 
	# 	save_file=result_dropbox_dir+"/test.pdf", interpolate=False, color='blue', 
	# 	linestyle='-.', 
	# 	marker=['o']*len(y), title=None)
	return unlbl_index[ids]



if __name__ == "__main__":
	FLAGS(sys.argv)
	get_full_os()
	ith_trials=range(1,20)
	show_performance_DU(ith_trials=ith_trials, # 2,3,4,5,
		dt="DU", is_relative=False)

	show_performance_OS(ith_trials=ith_trials, # 2,3,4,5,
		dt="OS", is_relative=False)












		