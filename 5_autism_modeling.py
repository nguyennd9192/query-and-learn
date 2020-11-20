from params import *
from utils.plot import *

import sys, pickle, functools, json, copy
from run_experiment import get_savedir, get_savefile, get_data_from_flags
import matplotlib.pyplot as plt
from utils.general_lib import *
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import joypy
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as grid_spec


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}


def load_Xy_query(unlbl_dir, unlbl_job, qid, unlbl_X, unlbl_y, unlbl_index, estimator_update_by):
	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	if qid == 1:
		selected_inds = []
		selected_inds_to_estimator = []
		all_query = None
	else:
		queried_idxes = range(1, qid)
		queried_files = [unlbl_dir + "/query_{}".format(k) + "/m0.1_c0.1.csv" for k in queried_idxes]
		print ("n queried files:", len(queried_files))
		valid_Xyid = get_queried_data(queried_files=queried_files, 
			database_results=database_results, 
			unlbl_X=unlbl_X, unlbl_index=unlbl_index,
			coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst)
		# # alocate representation, label and id of labeled data
		dq_X, dq_y, dq_idx = valid_Xyid[0]
		os_X, os_y, os_idx = valid_Xyid[1]
		rnd_X, rnd_y, rnd_idx = valid_Xyid[2]

		all_query = [[dq_X, dq_y, dq_idx], [os_X, os_y, os_idx], [rnd_X, rnd_y, rnd_idx]]
		# print ("DQ, OS, RND shape")
		assert dq_X.shape[0] == dq_y.shape[0]
		assert os_X.shape[0] == os_X.shape[0]
		assert rnd_X.shape[0] == rnd_X.shape[0]

		# # remove all labeled data of X, y, id to update sampler
		all_lbl_id = np.concatenate((dq_idx, os_idx, rnd_idx)).ravel()
		all_unlbl_y = np.concatenate((dq_y, os_y, rnd_y)).ravel()
		all_unlbl_X = np.concatenate((dq_X, os_X, rnd_X), axis=0)

		selected_inds = [np.where(unlbl_index==k)[0][0] for k in all_lbl_id]
		unlbl_y[selected_inds] = all_unlbl_y

		if estimator_update_by is not None:
			tmp = []
			if "DQ" in estimator_update_by:
				tmp.append(dq_idx)
			if "OS" in estimator_update_by:
				tmp.append(os_idx)
			if "RND" in estimator_update_by:
				tmp.append(rnd_idx)
			dt2estimator = np.concatenate(tmp).ravel()
			selected_inds_to_estimator = [np.where(unlbl_index==k)[0][0] for k in dt2estimator]
		else:
			selected_inds_to_estimator = copy.copy(selected_inds)



	# # this qid data
	this_qid_file = [unlbl_dir + "/query_{}".format(qid) + "/m0.1_c0.1.csv"]
	this_qid_Xy = get_queried_data(queried_files=this_qid_file, 
		database_results=database_results, 
		unlbl_X=unlbl_X, unlbl_index=unlbl_index,
		coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst)

	return unlbl_y, selected_inds, selected_inds_to_estimator, all_query, this_qid_Xy


def generate_verts(df_data, err_cols, save_fig):
	verts = []
	ax_objs = []

	n_panels = len(err_cols)
	gs = grid_spec.GridSpec(n_panels,1)
	fig = plt.figure(figsize=(16,9))

	dz = range(n_panels)
	cnorm = plt.Normalize()
	colors = plt.cm.jet(cnorm(dz))
	
	error_min = np.min(df_data.min().values)
	error_max = np.max(df_data.max().values) * 1.08
	print (error_min)

	for ind_time, col in enumerate(err_cols):
		X = df_data[col].values
		# X = np.abs(X[~np.isnan(X)])
		X = X[~np.isnan(X)]

		if np.max(X)==np.min(X):
			X_plot = X
		else:
			# X_plot = np.arange(np.min(X), np.max(X), (np.max(X)-np.min(X)) / 200.0)
			X_plot = np.arange(error_min, error_max, (error_max-error_min) / 200.0)
		

		parameters = norm.fit(X)
		kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(X.reshape(-1,1))
		log_dens = kde.score_samples(X_plot.reshape(-1,1))
		
		ys = np.exp(log_dens)
		ys[0], ys[-1] = 0.0, 0.0
		# verts.append(list(zip(X_plot, ys)))

		# creating new axes object
		ax_objs.append(fig.add_subplot(gs[n_panels - (ind_time+1):n_panels -ind_time, 0:]))
		# ax_objs.append(fig.add_subplot(gs[ind_time:ind_time+1, 0:]))

		# plotting the distribution
		ax_objs[-1].plot(X_plot, ys,color="black",lw=1)
		ax_objs[-1].fill_between(X_plot, ys, 
			alpha=0.6, color=colors[ind_time])


		# setting uniform x and y lims
		# ax_objs[-1].set_xlim(0,1.7)
		# ax_objs[-1].set_ylim(0,5)
		# ax_objs[-1].set_xscale('log')


		# make background transparent
		rect = ax_objs[-1].patch
		rect.set_alpha(0)

		# remove borders, axis ticks, and labels
		ax_objs[-1].set_yticklabels([])

		if ind_time == 0: # len(err_cols)-1
			ax_objs[-1].set_xlabel("Mean Absolute Error", fontsize=16,fontweight="bold")
		else:
			ax_objs[-1].set_xticklabels([])

		spines = ["top","right","left","bottom"]
		for s in spines:
			ax_objs[-1].spines[s].set_visible(False)
			ax_objs[-1].xaxis.set_ticks_position('none') 

		adj_country = col.replace("_"," ")
		# ax_objs[-1].text(-0.02,0,col,fontweight="bold",fontsize=14,ha="right")
		ax_objs[-1].text(error_min*1.03,0,col,fontweight="bold",fontsize=14,ha="right")

	gs.update(hspace=-0.6)

	fig.text(0.07,0.9,"Distribution of error in predicting whole screening space",fontsize=20)

	plt.tight_layout()
	# ax.set_title("error distribution")
	plt.savefig(save_fig, transparent=False)
	print("Save at:", save_fig)




def show_trace(ith_trial):
	unlbl_job = "mix" # mix, "mix_2-24"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 
	# unlbl_dir = result_file.replace(".pkl","")+"/"+unlbl_job


	# # load label and un_label
	X_trval, y_trval, index_trval, X_test, y_test, test_idx = get_data_from_flags()
	n_trval = len(X_trval)
	all_results = load_pickle(result_file)
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(
		unlbl_job=unlbl_job, result_file=result_file)
	
	# estimator_update_by = ["DQ"]
	estimator_update_by = None


	if FLAGS.score_method == "u_gp_mt":
		mt_kernel = 1.0 # 0.001, 1.0
		fix_update_coeff = 1
		unlbl_dir += "_mt{}".format(mt_kernel)
	
	elif FLAGS.score_method == "u_gp":
		if len(estimator_update_by) < 3:
			for k in estimator_update_by:
				unlbl_dir += k

	qids = range(1, 50)
	# qids = [1]
	eval_files = [unlbl_dir+"/query_{0}/eval_query_{0}.pkl".format(qid) for qid in qids]
	est_files = [unlbl_dir+"/query_{0}/pre_trained_est.pkl".format(qid) for qid in qids]

	flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
				  linestyle='none', markeredgecolor='k')
	
	error_rst_df = pd.DataFrame(index=unlbl_index, columns=["err_{}".format(k) for k in qids])
	var_rst_df = pd.DataFrame(index=unlbl_index, columns=["var_{}".format(k) for k in qids])
	
	error_save_at = unlbl_dir + "/autism/error.csv"
	var_save_at = unlbl_dir + "/autism/var.csv"
	save_fig = unlbl_dir + "/autism/mae_rest.pdf"
	
	makedirs(error_save_at)
	makedirs(var_save_at)
	
	color = "blue"

	all_error = []
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	tmp_df = pd.DataFrame(columns=["error", "qid"])

	for qid, eval_file, est_file in zip(qids, eval_files, est_files):
		# # get all_query, selected_inds and update unlbl_y
		unlbl_y, selected_inds, selected_inds_to_estimator, all_query, this_qid_Xy = load_Xy_query(
			unlbl_dir=unlbl_dir, unlbl_job=unlbl_job, qid=qid, 
			unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			estimator_update_by=estimator_update_by)

		_x_train, _y_train = est_alpha_updated(
				X_train=X_trval, y_train=y_trval, 
				X_test=unlbl_X, y_test=unlbl_y, 
				selected_inds=selected_inds_to_estimator)
		dq_X, dq_y, dq_idx = this_qid_Xy[0]
		os_X, os_y, os_idx = this_qid_Xy[1]
		rnd_X, rnd_y, rnd_idx = this_qid_Xy[2]
		
		print ("qid:", qid, "_x_train.shape", _x_train.shape)
		data = load_pickle(eval_file)
		estimator = load_pickle(est_file)
		estimator.fit(_x_train, _y_train)

		dict_values = data["DQ"]
		idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]
		
		rest_index = []
		for j, idx in enumerate(unlbl_index):
			if j not in selected_inds:
				rest_index.append(idx)
		filter_data = get_all_unlb_y(
				database_results=database_results, 
				unlbl_X=unlbl_X, unlbl_index=np.array(rest_index),
				coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst)
		unlbl_X_filter, unlbl_y_filter, unlbl_idx_filter = filter_data[0], filter_data[1], filter_data[2]

		X_test = copy.copy(unlbl_X_filter)
		y_test = copy.copy(unlbl_y_filter)
		idx_test = copy.copy(unlbl_idx_filter)

		y_pred = estimator.predict(X_test)
		r2 = r2_score(y_test, y_pred)
		mae = mean_absolute_error(y_test, y_pred)
		error = np.abs(y_test-y_pred)
		all_error.append( error )

		# # to plot

		bplot = ax.boxplot(x=error, vert=True, #notch=True, 
			sym='ro', # whiskerprops={'linewidth':2},
			positions=[qid], patch_artist=True,
			widths=0.1, meanline=True, flierprops=flierprops,
			showfliers=True, showbox=True, showmeans=False,
			autorange=True, bootstrap=5000)
		ax.text(qid, mae, round(mae, 2),
			horizontalalignment='center', size=14, 
			color="red", weight='semibold')
		patch = bplot['boxes'][0]
		patch.set_facecolor(color)



		indexes = ["{0}_{1}".format(k, qid) for k in unlbl_idx_filter]
		for i in range(len(indexes)):
			tmp_df.loc[indexes[i], "error"] = np.log(error[i])
			tmp_df.loc[indexes[i], "qid"] = qid



		# # end plot
		var = estimator.predict_proba(X_test)
		print ("len(X_test): ", len(X_test))

		var_rst_df.loc[idx_test, "var_{}".format(qid)] = var
		error_rst_df.loc[idx_test, "err_{}".format(qid)] = y_test - y_pred

		var_rst_df.to_csv(var_save_at)
		error_rst_df.to_csv(error_save_at)

		print ("n test:", len(y_test))
		print ("r2:", round(r2, 3), "mae:",  round(mae, 3))
		print ("var:", var)
		print ("=======")
		
		ax.grid(which='both', linestyle='-.')
		ax.grid(which='minor', alpha=0.2)
		plt.title(get_basename(save_fig))
		# ax.set_yscale('log')

		plt.savefig(save_fig, transparent=False)


	var_rst_df.fillna(0, inplace=True)
	error_rst_df.fillna(0, inplace=True)

	print (tmp_df)
	# ax = joypy.joyplot(tmp_df, by="qid", column="error")
	# ax.grid(which='both', linestyle='-.')
	# ax.grid(which='minor', alpha=0.2)
	# plt.title(get_basename(save_fig))
	# plt.savefig(save_fig.replace(".pdf","_joy.pdf"), transparent=False)

	plot_heatmap(matrix=var_rst_df.values, vmin=None, vmax=None, save_file=var_save_at.replace(".csv", ".pdf"), cmap="jet")
	plot_heatmap(matrix=error_rst_df.values, vmin=-0.5, vmax=0.5, save_file=error_save_at.replace(".csv", ".pdf"), cmap="bwr")



def error_dist(ith_trial):
	unlbl_job = "mix" # mix, "mix_2-24"
	# estimator_update_by = ["DQ"]
	estimator_update_by = None

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 

	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(
		unlbl_job=unlbl_job, result_file=result_file)
	if FLAGS.score_method == "u_gp_mt":
		mt_kernel = 0.001# 0.001, 1.0
		fix_update_coeff = 1
		unlbl_dir += "_mt{}".format(mt_kernel)
	elif FLAGS.score_method == "u_gp" and estimator_update_by is not None:
		if len(estimator_update_by) < 3:
			for k in estimator_update_by:
				unlbl_dir += k
	
	error_save_at = unlbl_dir + "/autism/error.csv"
	var_save_at = unlbl_dir + "/autism/var.csv"
	save_fig = unlbl_dir + "/autism/error_dist.pdf"


	qids = range(1, 50)
	err_cols = ["err_{}".format(qid) for qid in qids]

	error_rst_df = pd.read_csv(error_save_at, index_col=0)
	var_rst_df = pd.read_csv(var_save_at, index_col=0)


	generate_verts(error_rst_df, err_cols, save_fig)

	# fig = plt.figure(figsize=(10, 8))
	# ax = fig.add_subplot(1, 1, 1, projection='3d')
	# poly = PolyCollection(verts, facecolors="red", edgecolors="black")
	# poly.set_alpha(0.4)
	# ax.add_collection3d(poly, zs=qids, zdir='y')
	# # ax.set_yticklabels([""]+err_cols, verticalalignment='baseline',
	# # 	horizontalalignment='left') # , rotation=320
	# ax.set_xlim3d(0.001, 1.5)
	# ax.set_xlabel("error")
	# # ax.set_xscale('log')

	# ax.set_ylim3d(-0.5, len(err_cols) + 0.05)
	# ax.set_ylabel("Query times")
	# ax.set_zlim3d(0, 10)
	# ax.grid(False)
	# ax.w_xaxis.pane.fill = False
	# ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
	# ax.w_yaxis.pane.fill = False
	# ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
	# ax.view_init(30, 320)
	# ax.zaxis.set_visible(False)
	# ax.set_title("error distribution")
	# plt.savefig(save_fig, transparent=False)
	# # plt.show()
	# print("Save at:", save_fig)


if __name__ == "__main__":
	FLAGS(sys.argv)
	is_label_mix = False
	for sm in ["margin"]: # "uniform",  "exploitation", "expected_improvement", "margin"
		FLAGS.sampling_method = sm
		show_trace(ith_trial="000")
		error_dist(ith_trial="000")

	# # to label the "mix" job
	if is_label_mix:
		unlbl_csv = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/unlabeled_data/mix.csv"
		vasp_lbl2mix(unlbl_file=unlbl_csv, 
			database_results=database_results, 
			coarse_db_rst=coarse_db_rst, 
			fine_db_rst=fine_db_rst)




