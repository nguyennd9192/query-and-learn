from params import *
from utils.plot import *

import sys, pickle, functools, json, copy
from run_experiment import get_savedir, get_savefile, get_data_from_flags
import matplotlib.pyplot as plt
from utils.general_lib import *
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}



def load_Xy_query(unlbl_dir, unlbl_job, qid, unlbl_X, unlbl_y, unlbl_index):
	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	if qid == 1:
		selected_inds = []
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


	# # this qid data
	this_qid_file = [unlbl_dir + "/query_{}".format(qid) + "/m0.1_c0.1.csv"]
	this_qid_Xy = get_queried_data(queried_files=this_qid_file, 
		database_results=database_results, 
		unlbl_X=unlbl_X, unlbl_index=unlbl_index,
		coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst)

	return unlbl_y, selected_inds, all_query, this_qid_Xy

def show_trace(ith_trial):
	unlbl_job = "mix" # mix, "mix_2-24"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 
	unlbl_dir = result_file.replace(".pkl","")+"/"+unlbl_job

	# # load label and un_label
	X_trval, y_trval, index_trval, X_test, y_test, test_idx = get_data_from_flags()
	n_trval = len(X_trval)
	all_results = load_pickle(result_file)
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(
		unlbl_job=unlbl_job, result_file=result_file)


	qids = range(1, 11)
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
	for qid, eval_file, est_file in zip(qids, eval_files, est_files):
		# # get all_query, selected_inds and update unlbl_y
		unlbl_y, selected_inds, all_query, this_qid_Xy = load_Xy_query(
			unlbl_dir=unlbl_dir, unlbl_job=unlbl_job, qid=qid, 
			unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index)

		_x_train, _y_train = est_alpha_updated(
				X_train=X_trval, y_train=y_trval, 
				X_test=unlbl_X, y_test=unlbl_y, 
				selected_inds=selected_inds)
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
			# sym='rs', # whiskerprops={'linewidth':2},
			positions=[qid], patch_artist=True,
			widths=0.1, meanline=True, flierprops=flierprops,
			showfliers=False, showbox=True, showmeans=False)
		ax.text(qid, mae, round(mae, 2),
			horizontalalignment='center', size=14, 
			color="red", weight='semibold')
		patch = bplot['boxes'][0]
		patch.set_facecolor(color)
		# # end plot
		var = estimator.predict_proba(X_test)

		var_rst_df.loc[idx_test, "var_{}".format(qid)] = var
		error_rst_df.loc[idx_test, "err_{}".format(qid)] = y_test - y_pred

		var_rst_df.to_csv(var_save_at)
		error_rst_df.to_csv(error_save_at)

		print ("n test:", len(y_test))
		print ("r2:", round(r2, 3), "mae:",  round(mae, 3))
		print ("var:", var)
		print ("=======")
		
		ax.set_yscale('log')
		ax.grid(which='both', linestyle='-.')
		ax.grid(which='minor', alpha=0.2)


		plt.title(get_basename(save_fig))
		plt.savefig(save_fig, transparent=False)
		print ("Save at: ", save_fig)

	var_rst_df.fillna(0, inplace=True)
	error_rst_df.fillna(0, inplace=True)

	plot_heatmap(matrix=var_rst_df.values, vmin=None, vmax=None, save_file=var_save_at.replace(".csv", ".pdf"), cmap="jet")
	plot_heatmap(matrix=error_rst_df.values, vmin=-0.5, vmax=0.5, save_file=error_save_at.replace(".csv", ".pdf"), cmap="bwr")




	# for p, error in enumerate(all_error):
	# 	bplot = ax.boxplot(x=error, vert=True, #notch=True, 
	# 		# sym='rs', # whiskerprops={'linewidth':2},
	# 		positions=[p], patch_artist=True,
	# 		widths=0.1, meanline=True, flierprops=flierprops,
	# 		showfliers=False, showbox=True, showmeans=False)
	# 	# ax.text(pos_x, mean, round(mean, 2),
	# 	# 	horizontalalignment='center', size=14, 
	# 	# 	color=color, weight='semibold')
	# 	patch = bplot['boxes'][0]
	# 	patch.set_facecolor("blue")
	# 	# for dt, dict_values in data.items():
	# 	# 	idx_qr, y_qr, y_qr_pred = dict_values["idx_qr"], dict_values["y_qr"], dict_values["y_qr_pred"]


if __name__ == "__main__":
	FLAGS(sys.argv)
	for sm in [ "margin" ]: # "uniform", "exploitation", "expected_improvement",
		FLAGS.sampling_method = sm
		show_trace(ith_trial="000")
