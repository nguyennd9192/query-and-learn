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
from sklearn.metrics import pairwise 


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}


def load_Xy_query(unlbl_dir, qid, 
	unlbl_X, unlbl_y, unlbl_index, estimator_update_by):
	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	if qid == 1:
		selected_inds = []
		selected_inds_to_estimator = []
		all_query = None
	else:
		queried_idxes = range(1, qid)
		queried_files = [unlbl_dir + "/query_{}".format(k) + "/query.csv" for k in queried_idxes]
		print ("n queried files:", len(queried_files))
		# # query only original ofm, transform later
		valid_Xyid = get_queried_data(queried_files=queried_files, 
			database_results=database_results, 
			unlbl_X=unlbl_X, unlbl_index=unlbl_index,
			coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst,
			embedding_model="org_space")


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
	this_qid_file = [unlbl_dir + "/query_{}".format(qid) + "/query.csv"]
	this_qid_Xy = get_queried_data(queried_files=this_qid_file, 
		database_results=database_results, 
		unlbl_X=unlbl_X, unlbl_index=unlbl_index,
		coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst,
		embedding_model="org_space")

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
	result_dir, unlbl_dir = get_savedir()
	X_trval, y_trval, index_trval, unlbl_X, unlbl_y, unlbl_index = get_data_from_flags()


	n_unlbl = len(unlbl_index)
	
	

	qids = range(1, FLAGS.n_run)
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

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	tmp_df = pd.DataFrame(columns=["error", "qid"])

	color_array = []

	line_query = []



	for qid, eval_file, est_file in zip(qids, eval_files, est_files):

		# # get all_query, selected_inds and update unlbl_y
		unlbl_y, selected_inds, selected_inds_to_estimator, all_query, this_qid_Xy = load_Xy_query(
			unlbl_dir=unlbl_dir, qid=qid, 
			unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			estimator_update_by=estimator_update_by)

		# _x_train, _y_train = est_alpha_updated(
		# 		X_train=X_trval, y_train=y_trval, 
		# 		X_test=unlbl_X, y_test=unlbl_y, 
		# 		selected_inds=selected_inds_to_estimator)
		estimator = load_pickle(est_file)

		# # _x_train = X_trval[selected_inds_to_estimator]
		# # all _x_train, _y_train, _unlbl_X have been transformed if needed
		_x_train, _y_train, unlbl_X_sampler, embedding_model = est_alpha_updated(
			X_train=X_trval, y_train=y_trval, 
			X_test=unlbl_X, y_test=unlbl_y, 
			selected_inds=selected_inds_to_estimator,
			embedding_method=FLAGS.embedding_method,
			mae_update_threshold=FLAGS.mae_update_threshold,
			estimator=estimator) 


		dq_X, dq_y, dq_idx = this_qid_Xy[0]
		os_X, os_y, os_idx = this_qid_Xy[1]
		rnd_X, rnd_y, rnd_idx = this_qid_Xy[2]
		
		print ("qid:", qid, "_x_train.shape", _x_train.shape)
		print ("mae_update_threshold", FLAGS.mae_update_threshold)
		data = load_pickle(eval_file)
		# estimator.fit(_x_train, _y_train)

		


		assert unlbl_X_sampler.shape[0] == unlbl_X.shape[0]
		_unlbl_dist = pairwise.euclidean_distances(unlbl_X_sampler)
		metric_df = pd.DataFrame(_unlbl_dist, index=unlbl_index, columns=unlbl_index)

		save_file = save_file = unlbl_dir+"/query_{0}/{1}_dist.png".format(qid, FLAGS.embedding_method)
		metric_df.to_csv(save_file.replace(".png", ".csv"))
		
		line_query.append(len(selected_inds))
		plot_heatmap(matrix=metric_df.values, 
				vmin=None, vmax=None, save_file=save_file, 
				cmap="jet", title=save_file.replace(ALdir, ""),
				lines=line_query)


		if False:	
			# # plot only embedding methods
			if _x_train.shape[1] == 2:
				save_file = unlbl_dir+"/query_{0}/{1}".format(qid, FLAGS.embedding_method)
				new_c = [qid] * (_x_train.shape[0] - len(color_array))
				color_array += new_c
				scatter_plot_2(x=_x_train[:, 0], y=_x_train[:, 1], 
					color_array=color_array, xvline=None, yhline=None, 
					sigma=None, mode='scatter', lbl=None, name=None, 
					x_label='x', y_label='y', 
					save_file=save_file, interpolate=False, color='blue', 
					preset_ax=None, linestyle='-.', marker='o')



			y_pred = estimator.predict(X_test)
			y_test, y_pred = filter_array(y_test, y_pred)


			assert y_test.shape == y_pred.shape
			r2 = r2_score(y_test, y_pred)
			mae = mean_absolute_error(y_test, y_pred)
			error = np.abs(y_test-y_pred)

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



			indexes = ["{0}_{1}".format(k, qid) for k in unlbl_idx]
			for i in range(len(indexes)):
				tmp_df.loc[indexes[i], "error"] = np.log(error[i])
				tmp_df.loc[indexes[i], "qid"] = qid


			# # end plot
			var = estimator.predict_proba(X_test)

			var_rst_df.loc[idx_test, "var_{}".format(qid)] = var
			error_rst_df.loc[idx_test, "err_{}".format(qid)] = y_test - y_pred

			var_rst_df.to_csv(var_save_at)
			error_rst_df.to_csv(error_save_at)
			print ("r2:", round(r2, 3), "mae:",  round(mae, 3))
			print ("=======")
			
			ax.grid(which='both', linestyle='-.')
			ax.grid(which='minor', alpha=0.2)
			plt.title(unlbl_dir.replace(ALdir, ""))
			# ax.set_yscale('log')

			plt.savefig(save_fig, transparent=False)

			var_rst_df.fillna(0, inplace=True)
			error_rst_df.fillna(0, inplace=True)

			# ax = joypy.joyplot(tmp_df, by="qid", column="error")
			# ax.grid(which='both', linestyle='-.')
			# ax.grid(which='minor', alpha=0.2)
			# plt.title(get_basename(save_fig))
			# plt.savefig(save_fig.replace(".pdf","_joy.pdf"), transparent=False)

			plot_heatmap(matrix=var_rst_df.values, vmin=None, vmax=None, save_file=var_save_at.replace(".csv", ".pdf"), cmap="jet")
			plot_heatmap(matrix=error_rst_df.values, vmin=-0.5, vmax=0.5, save_file=error_save_at.replace(".csv", ".pdf"), cmap="bwr")



def error_dist(ith_trial):
	# estimator_update_by = None
	result_dir, unlbl_dir = get_savedir()
	X_trval, y_trval, index_trval, unlbl_X, unlbl_y, unlbl_index = get_data_from_flags()


	estimator_update_by = ["DQ"] # , "RND", "OS"
	if len(estimator_update_by) < 3:
		for k in estimator_update_by:
			unlbl_dir += k

	if FLAGS.score_method == "u_gp_mt":
		mt_kernel = 0.001 # 0.001, 1.0
		fix_update_coeff = 1
		unlbl_dir += "_mt{}".format(mt_kernel)

	
	error_save_at = unlbl_dir + "/autism/error.csv"
	var_save_at = unlbl_dir + "/autism/var.csv"
	save_fig = unlbl_dir + "/autism/error_dist.pdf"

	error_rst_df = pd.read_csv(error_save_at, index_col=0)
	var_rst_df = pd.read_csv(var_save_at, index_col=0)

	error_rst_df = error_rst_df.dropna(axis="columns", how="all")
	var_rst_df = var_rst_df.dropna(axis="columns", how="all")
	err_cols = error_rst_df.columns

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



# def rotate_matrix(df_matrix):
# 	values_matrix = df_matrix.values
# 	features = df_matrix.columns.values
# 	values_matrix = np.rot90(values_matrix)
# 	df_result = pd.DataFrame(values_matrix, columns=features)
# 	df_result.index = features[::-1]
# 	return df_result

# def plot_heatmap(df_matrix, title, fig_name, output_dir="", ivl=None, df_mapping=None):
# 	if not ivl is None:
# 		tmp_data = []
# 		for label in ivl:
# 			tmp_data.append(df_matrix.loc[label])
# 		optimize_df = pd.DataFrame(tmp_data)
# 		optimize_df = optimize_df.reindex(ivl, axis=1)
# 		optimize_df = rotate_matrix(optimize_df)




if __name__ == "__main__":
	FLAGS(sys.argv)
	
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	FLAGS.score_method = kwargs["score_method"]
	FLAGS.sampling_method =	kwargs["sampling_method"]
	FLAGS.embedding_method = kwargs["embedding_method"]

	show_trace(ith_trial="000")
	# error_dist(ith_trial="000")



	# is_label_mix = False
	# for sm in ["margin"]: # "uniform",  "exploitation", "expected_improvement", "margin"
	# 	FLAGS.sampling_method = sm
	# 	show_trace(ith_trial="000")
	# 	error_dist(ith_trial="000")

	# # # to label the "mix" job
	# if is_label_mix:
	# 	unlbl_csv = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/unlabeled_data/mix.csv"
	# 	vasp_lbl2mix(unlbl_file=unlbl_csv, 
	# 		database_results=database_results, 
	# 		coarse_db_rst=coarse_db_rst, 
	# 		fine_db_rst=fine_db_rst)




