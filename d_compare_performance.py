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

# performance_codes = dict({"org_space":"blue", "MLKR":"red"}) 
performance_codes = dict({
	"org_space|uniform":"cornflowerblue", "org_space|margin":"cornflowerblue", "org_space|exploitation":"cornflowerblue", 

	"MLKR|uniform":"darkred", "MLKR|margin":"darkred", "MLKR|exploitation":"darkred", 
	}) 

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

	return error_qr, error_non_qr, data
	# return mean_qr, mean_non_qr, data

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
	job_savedir = get_savedir(ith_trial=ith_trials[0])

	mean_trials = []
	var_trials = []
	patch = None
	for qid in qids:
		values = []
		# # for all trials
		n_trials = 0
		for ith_trial in ith_trials:
			# try:
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
					this_recall = len(list(set(idx_expl).intersection(full_os_ids))) / float(n_full)
					recall_os += this_recall
				values.append(recall_os)
			elif dt == "DU":
				values.append(mean_non_qr)
				# values = np.concatenate([values, mean_non_qr])
			else:
				values.append(mean_qr)
				# values = np.concatenate([values, mean_qr])

			n_trials += 1
			# except Exception as e:
			# 	print ("Error in ", embedding_method, sampling_method, qid, ith_trial)
			# 	pass
		values = np.array(values)
		if dt == "OS":
			# # normalize in scale of outstanding 
			values /= n_trials
		# else:
		# 	# # normalize in scale of range of formation energy
		# 	values /= 0.43

		mean_trial = np.mean(values)
		var_trial = np.var(values)

		if dt == "DU": # OS
			lab = "{0}|{1}".format(embedding_method, sampling_method)

			# bplot = ax.boxplot(x=np.ravel(values), vert=True, #notch=True, 
			# 		# sym='rs', # whiskerprops={'linewidth':2},
			# 		# alpha=0.4,
			# 		# notch=True,
			# 		positions=[qid], patch_artist=False,
			# 		widths=0.8, meanline=False, #flierprops=flierprops,
			# 		showfliers=False, #showbox=True, 
			# 		showmeans=True,
			# 		)
			# 	# ax.text(pos_x, mean, round(mean, 2),
			# 	# 	horizontalalignment='center', size=14, 
			# 	# 	color=color, weight='semibold')
			# patch = bplot['boxes'][0]
			# patch.set_facecolor(performance_codes[lab])
			# # patch.set_hatch(hatch_codes[sampling_method])
			# patch.set_alpha(0.8)
			# ax.legend([bplot["boxes"][0]], 
			# 	["{0}_{1}".format(embedding_method, sampling_method)], 
			# 	loc='upper right')

			# # to plot violin
			data = np.ravel(values)
			violin_parts = ax.violinplot(dataset=data, positions=[qid], # 
						showmeans=True, vert=True, #showmedians=True
						showextrema=False,  # False
						points=len(data)
						)
			set_color_violin(x_flt=data, violin_parts=violin_parts, 
				pc=performance_codes[lab], nc="darkblue")


		

		mean_trials.append(mean_trial)
		var_trials.append(var_trial)
	return np.array(mean_trials), np.array(var_trials), patch



def show_performance(ith_trials, dt, is_relative):
	performance_pos = dict({"uniform":0, "exploitation":1, 
				"margin":2, "expected_improvement":3})
	fig = plt.figure(figsize=(11, 8))
	ax = fig.add_subplot(1, 1, 1)
	# ax.grid(which='both', linestyle='-.')
	# ax.grid(which='minor', alpha=0.2)


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
			# patches.append(patch)
			# if dt != "OS":
			# ax.plot(qids, mean_trials, '-', c=performance_codes[lab], 
			# 	label=lab, linestyle=linestyle_dict[sampling_method], linewidth=2)
			# ax.fill_between(qids, mean_trials - var_trials, mean_trials + var_trials, 
			# 	color=performance_codes[lab],
			# 	alpha=0.8)
			# print (var_trials)

			# except:
			# 	pass
		# ax.set_xlabel(r"Query index", **axis_font) 
		if dt == "OS":
			ax.set_ylabel(r"Recall rate", **axis_font)
			# ax.set_ylim(-1.6, -0.4)
			# ax.set_yscale('log')

		else:
			ax.set_ylabel(r"Error (eV/atom)", **axis_font)
			# ax.set_ylim(0.0, 0.04)
			# ax.set_yscale('log')
			# ax.set_ylim(0.006, 0.04)
			print ("Here!!!")
	# ax.legend(patches, legends)
	ax.legend()
	
	plt.xticks(qids, qids) 
	ax.tick_params(axis='y', labelsize=12)
	ax.set_title(dt)
	# ax.set_xlim(1.0, n_run)


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
	# get_full_os()

	for dt in ["DU"]: # "DQ", "OS", "RND", "DQ_to_RND", "DU"
		show_performance(ith_trials=range(11,31), # 2,3,4,5,
			# 2
			dt=dt, is_relative=False)

		