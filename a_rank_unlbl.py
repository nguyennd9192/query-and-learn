

import sys, pickle, functools, json, copy, random, re
import numpy as np 
from params import *
from absl import app
from utils.utils import load_pickle
from utils.general_lib import *
 
from utils import utils
from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping

from sklearn.metrics import pairwise 

from proc_results import read_exp_params, params2text

from utils.manifold_processing import Preprocessing
from utils.plot import *
from tensorflow.io import gfile

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvas
import cv2 as cv
from scipy.interpolate import griddata
from preprocess_data import read_deformation
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support

from itertools import product
from embedding_space import InversableEmbeddingSpace



import warnings
warnings.filterwarnings("ignore")

get_wrapper_AL_mapping()


def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
									 **kwargs):
		n_active = int(mixture * N)
		n_passive = N - n_active
		kwargs["N"] = n_active
		kwargs["already_selected"] = already_selected
		batch_AL, acq_val = sampler.select_batch(**kwargs)
		already_selected = already_selected + batch_AL
		kwargs["N"] = n_passive
		kwargs["already_selected"] = already_selected
		# kwargs_copy = copy.copy(kwargs)
		# if "return_best_sim" in kwargs_copy.keys():
		#   del kwargs_copy[key]
		# batch_PL = uniform_sampler.select_batch(**kwargs_copy)
		batch_PL, p = uniform_sampler.select_batch(**kwargs)
		return batch_AL + batch_PL, acq_val

def process_dimensional_reduction(unlbl_X, method):
	processing = Preprocessing()
	if method == "tsne":
		config_tsne = dict({"n_components":2, "perplexity":500.0,  # same meaning as n_neighbors
			"early_exaggeration":1000.0, # same meaning as n_cluster
			"learning_rate":1000.0, "n_iter":1000,
			 "n_iter_without_progress":300, "min_grad_norm":1e-07, 
			 "metric":'euclidean', "init":'random',
			 "verbose":0, "random_state":None, "method":'barnes_hut', 
			 "angle":0.5, "n_jobs":None})
		processing.similarity_matrix = unlbl_X
		X_trans, _, a, b = processing.tsne(**config_tsne)

	if method == "mds":
		config_mds = dict({"n_components":2, "metric":True, "n_init":4, 
			"max_iter":300, "verbose":0,
				"eps":0.001, "n_jobs":None, "random_state":None, 
				"dissimilarity":'precomputed'})
		cosine_distance = 1 - pairwise.cosine_similarity(unlbl_X)
		processing.similarity_matrix = cosine_distance
		X_trans, _ = processing.mds(**config_mds)

	if method == "isomap":
		config_isomap = dict({"n_neighbors":5, "n_components":2, 
			"eigen_solver":'auto', "tol":0, 
			"max_iter":None, "path_method":'auto',
			"neighbors_algorithm":'auto', 
			"n_jobs":None,
			"metric":"l1"})
		processing.similarity_matrix = unlbl_X
		X_trans, a, b = processing.iso_map(**config_isomap)
	return X_trans


def query_and_learn(FLAGS, qid, 
		selected_inds, selected_inds_to_estimator,
		estimator, X_train, y_train, index_train, 
		unlbl_X, unlbl_y, unlbl_index, sampler, uniform_sampler, 
		is_save_query, savedir, tsne_file, is_plot, perform_ax, perform_fig):
	# is_load_pre_trained = False
	csv_saveat = savedir+"/query_{0}/query_{0}.csv".format(qid)	
	fig_saveat = savedir + "/autism/error_dist.pdf"
	makedirs(csv_saveat)
	makedirs(fig_saveat)

	selected_inds_copy = copy.copy(selected_inds)
	
	query_data = dict()
	if is_save_query:
		last_query = np.array([None] * len(unlbl_index))
		last_query[selected_inds] = "last_query_{}".format(qid)
		query_data["last_query_{}".format(qid)] = last_query


	query_data["unlbl_index"] = unlbl_index
	# # end save querying data 


	# # update train, test by selected inds
	_x_train, _y_train, _unlbl_X, embedding_model = est_alpha_updated(
		X_train=X_train, y_train=y_train, 
		X_test=unlbl_X, y_test=unlbl_y, 
		selected_inds=selected_inds_to_estimator,
		estimator=estimator) # # in the past: selected_inds (update by all database)

	# # fit with whole
	# estimator.estimator = None # # force grid-search cv
	estimator.fit(_x_train, _y_train)

	unlbl_y_pred = estimator.predict(_unlbl_X)
	query_data["unlbl_y_pred_{}".format(qid)] = unlbl_y_pred

	select_batch_inputs = {"model": copy.copy(estimator), "labeled": None, 
			"eval_acc": None, "X_test": None, 
			"y_test": None, "y": None, "verbose": True,
			"y_star": min(_y_train)}

	# # 1. update by D_{Q}
	new_batch, acq_val = select_batch(sampler=sampler, uniform_sampler=uniform_sampler, 
						mixture=FLAGS.active_p, N=FLAGS.batch_size,
						already_selected=list(selected_inds_copy), **select_batch_inputs)
	selected_inds_copy.extend(new_batch)
	if is_save_query:
		query2update_DQ = np.array([None] * len(unlbl_index))
		query2update_DQ[new_batch] = "query2update_DQ_{}".format(qid)
		query_data["query2update_DQ_{}".format(qid)] = query2update_DQ
		query_data["acq_val_{}".format(qid)] = acq_val


	# # 2. select by D_{o/s}
	argsort_y_pred = np.argsort(unlbl_y_pred)
	outstand_idx = [k for k in argsort_y_pred if k not in selected_inds_copy]
	assert outstand_idx != []
	outstand_list = outstand_idx[:FLAGS.batch_outstand]
	lim_outstand_list = max(unlbl_y_pred[outstand_list])

	if is_save_query:
		query_outstanding = np.array([None] * len(unlbl_index))
		query_outstanding[outstand_list] = "query_outstanding_{}".format(qid)
		query_data["query_outstanding_{}".format(qid)] = query_outstanding

	selected_inds_copy.extend(outstand_list)
	max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

	# # 3. select by D_{rand}
	the_non_qr = list(set(range(_unlbl_X.shape[0])) - set(selected_inds_copy))
	random_list = random.sample(the_non_qr, FLAGS.batch_rand)
	selected_inds_copy.extend(random_list)

	if is_save_query:
		query_random = np.array([None] * len(unlbl_index))
		query_random[random_list] = "query_random_{}".format(qid)
		query_data["query_random_{}".format(qid)] = query_random

	# # measure error of all others
	assert unlbl_y_pred.shape == unlbl_y.shape

	error = np.abs(unlbl_y - unlbl_y_pred)	
	# # end plot
	var = estimator.predict_proba(_unlbl_X)
	if is_save_query:
		query_data["var_{}".format(qid)] = var
		query_data["err_{}".format(qid)] = error
		query_df = pd.DataFrame().from_dict(query_data)
		makedirs(csv_saveat)
		query_df.to_csv(csv_saveat)


	# # test error
	# # selected_inds_to_estimator or selected_inds
	non_qr_ids = list(set(range(unlbl_X.shape[0])) - set(selected_inds))
	non_qr_y_pred = unlbl_y_pred[non_qr_ids]
	non_qr_y = unlbl_y[non_qr_ids]
	non_qr_error = np.abs(non_qr_y - non_qr_y_pred)	

	# r2 = r2_score(non_qr_y, non_qr_y_pred)
	mae = mean_absolute_error(non_qr_y, non_qr_y_pred)

	flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
				  linestyle='none', markeredgecolor='k')
	bplot = perform_ax.boxplot(x=non_qr_error, vert=True, #notch=True, 
		sym='ro', # whiskerprops={'linewidth':2},
		positions=[qid], patch_artist=True,
		widths=0.1, meanline=True, flierprops=flierprops,
		showfliers=True, showbox=True, showmeans=False,
		autorange=True, bootstrap=5000)
	perform_ax.text(qid, mae, round(mae, 2),
		horizontalalignment='center', size=14, 
		color="red", weight='semibold')
	patch = bplot['boxes'][0]
	patch.set_facecolor("blue")

	perform_ax.grid(which='both', linestyle='-.')
	perform_ax.grid(which='minor', alpha=0.2)
	plt.title(fig_saveat.replace(ALdir, ""))
	plt.savefig(fig_saveat, transparent=False)


	plt_mode = "2D" # 3D, 2D, 3D_patch
	# # AL points ~ smallest min margin ~ biggest apparent points
	if FLAGS.sampling_method == "margin":
		acq_val[np.isinf(acq_val)] = np.max(acq_val)

	scaler = MinMaxScaler()
	size_points = scaler.fit_transform(acq_val.reshape(-1, 1))
	# # name, color, marker for plot
	plot_index = np.concatenate((unlbl_index, index_train), axis=0)
	name = [k.replace(ALdir, "") for k in plot_index]
	family = [get_family(k) for k in plot_index]

	list_cdict = np.array([get_color_112(k) for k in name])
	marker_array = np.array([get_marker_112(k) for k in family])
	alphas = np.array([0.3] * len(plot_index))
	alphas[selected_inds_copy] = 1.0 
	alphas[len(unlbl_index):] = 1.0

	# # plot MLKR space
	if FLAGS.embedding_method != "org_space":
		# # concatenate data points train test
		xy = np.concatenate((_unlbl_X, _x_train), axis=0)
		# # selected array as +
		ytrain_pred = estimator.predict(_x_train)

		y_all_pred = np.concatenate((unlbl_y_pred, ytrain_pred), axis=0)
		y_all_obs = np.concatenate((unlbl_y, _y_train), axis=0)
		error_all = y_all_pred - y_all_obs

		# # merge var all
		var_train = estimator.predict_proba(_x_train)
		var_all = np.concatenate((var, var_train), axis=0)  
		this_fig_dir = csv_saveat.replace(".csv", "ipl.pdf")
		marker_array[non_qr_ids] = "+"


		data = dict()
		data["x_embedd"] = xy[:, 0]
		data["y_embedd"] = xy[:, 1]
		data["index"] = plot_index
		data["y_obs"] = y_all_obs
		data["y_pred"] = y_all_pred
		data["error"] = error_all
		data["var"] = var_all
		data["marker"] = marker_array
		plt_df = pd.DataFrame().from_dict(query_data)
		this_df_dir = this_fig_dir.replace(".pdf", ".csv")
		makedirs(this_df_dir)
		plt_df.to_csv(this_df_dir)


		scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
			z_values=error_all,
			list_cdict=list_cdict, 
			xvlines=[0.0], yhlines=[0.0], 
			sigma=None, mode='scatter', lbl=None, name=None, 
			s=60, alphas=alphas, 
			title=None,
			x_label=FLAGS.embedding_method + "_dim_1",
			y_label=FLAGS.embedding_method + "_dim_2", 
			save_file=this_fig_dir.replace(".pdf", "_error.pdf"),
			interpolate=False, 
			preset_ax=None, linestyle='-.', marker=marker_array)

		scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
			z_values=y_all_obs,
			list_cdict=list_cdict, 
			xvlines=[0.0], yhlines=[0.0], 
			sigma=None, mode='scatter', lbl=None, name=None, 
			s=60, alphas=alphas, 
			title=None,
			x_label=FLAGS.embedding_method + "_dim_1",
			y_label=FLAGS.embedding_method + "_dim_2", 
			save_file=this_fig_dir.replace(".pdf", "_yobs.pdf"),
			interpolate=False, 
			preset_ax=None, linestyle='-.', marker=marker_array)

		scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
			z_values=y_all_pred,
			list_cdict=list_cdict, 
			xvlines=[0.0], yhlines=[0.0], 
			sigma=None, mode='scatter', lbl=None, name=None, 
			s=60, alphas=alphas, 
			title=None,
			x_label=FLAGS.embedding_method + "_dim_1",
			y_label=FLAGS.embedding_method + "_dim_2", 
			save_file=this_fig_dir.replace(".pdf", "_ypred.pdf"),
			interpolate=False, 
			preset_ax=None, linestyle='-.', marker=marker_array)

		scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
			z_values=var_all,
			list_cdict=list_cdict, 
			xvlines=[0.0], yhlines=[0.0], 
			sigma=None, mode='scatter', lbl=None, name=None, 
			s=60, alphas=alphas, 
			title=None,
			x_label=FLAGS.embedding_method + "_dim_1",
			y_label=FLAGS.embedding_method + "_dim_2", 
			save_file=this_fig_dir.replace(".pdf", "_yvar.pdf"),
			interpolate=False, 
			preset_ax=None, linestyle='-.', marker=marker_array)




		# lim_acq_val = min(acq_val[new_batch])

		# z =  np.concatenate((unlbl_y_pred, y_train)) # unlbl_y_pred, min_margin
		# x1 = np.arange(min(x), max(x), (max(x) - min(x))/200)
		# y1 = np.arange(min(y), max(y), (max(y) - min(x))/200)
		# xi, yi = np.meshgrid(x1,y1)
		# # interpolate
		# zi = griddata((x,y),z,(xi,yi),method='neanon_qr')
		# if plt_mode == "3D_patch":
		# 	xi, yi, zi = x, y, z
		# ax = ax_surf(xi=xi, yi=yi, zi=zi, label="pred_val", mode=plt_mode)

		# # # tSNE map
		# csv_save_dir += "/"+plt_mode
		# save_figat = csv_save_dir+"/cmap_unlbl_rank_unlbl_y_pred.pdf"
		# ax_scatter(ax=ax, x=x, y=y, marker=marker_array, list_cdict=list_cdict,
		# 	 x_label="tSNE axis 1", y_label="tSNE axis 2",
		# 	 alphas=alphas, save_at=save_figat, plt_mode=plt_mode)
		
		# # # new plot
		# ax2 = ax_surf(xi=xi, yi=yi, zi=zi, label="pred_val", mode=plt_mode)
		# list_cdict2 = np.array(copy.copy(list_cdict))
		# marker_array2 = np.array(copy.copy(marker_array))
		# mask = np.full(len(list_cdict2),False,dtype=bool)
		# mask[selected_inds_copy] = True # # for selected ids
		# mask[-len(index_train):] = True # # for obs dataset
		# list_cdict2[~mask] = dict({"grey":"full"})
		# marker_array2[~mask] = "o"

		# ax_scatter(ax=ax2, x=x, y=y, marker=marker_array2, 
		# 	list_cdict=list_cdict2,
		# 	x_label="tSNE axis 1", y_label="tSNE axis 2",
		# 	alphas=alphas, plt_mode=plt_mode,
		# 	save_at=save_figat.replace(".pdf", "2.pdf"))
		
		# # # acp_val map
		# lim_acq_val = min(acq_val[new_batch])
		# z =  np.concatenate((acq_val, [0]*len(y_train))) # unlbl_y_pred, min_margin
		# x1 = np.arange(min(x), max(x), (max(x) - min(x))/200)
		# y1 = np.arange(min(y), max(y), (max(y) - min(x))/200)
		# xi, yi = np.meshgrid(x1,y1)
		
		# # interpolate
		# zi = griddata((x,y),z,(xi,yi),method='neanon_qr')
		# if plt_mode == "3D_patch":
		# 	xi, yi, zi = x, y, z

		# ax = ax_surf(xi=xi, yi=yi, zi=zi, 
		# 	label="acp_val", mode=plt_mode)
		# # plt.show()

		# # # tSNE map
		# ax_scatter(ax=ax, x=x, y=y, marker=marker_array, list_cdict=list_cdict,
		# 	 x_label="tSNE axis 1", y_label="tSNE axis 2",
		# 	 alphas=alphas, plt_mode=plt_mode,
		# 	 save_at=save_figat.replace("unlbl_y_pred", "acq_val"))
		# try:
		# 	scatter_plot_5(x=acq_val, y=unlbl_y_pred, list_cdict=list_cdict, 
		# 		xvlines=[lim_acq_val], yhlines=[lim_outstand_list], 
		# 		sigma=None, mode='scatter', lbl=None, name=None, 
		# 		s=80, alphas=alphas, title=None,
		# 		x_label=sampler.name, y_label='unlbl_y_pred', 
		# 		save_file=save_figat.replace(".pdf", "_2.pdf"),
		# 		interpolate=False, 
		# 		preset_ax=None, linestyle='-.', marker=marker_array)
		# except Exception as e:
		# 	pass

	return _x_train, _y_train, estimator, embedding_model


def evaluation_map(FLAGS, X_train, y_train, index_train, 
	all_query, sampler, uniform_sampler, 
	save_at, eval_data_file, estimator):
	"""
	# # to create an error map of samples in each query batch
	"""
	DQ, OS, RND = all_query 
	all_query_name = ["DQ", "OS", "RND"]
	feedback_val = None
	fig = plt.figure(figsize=(10, 8))
	grid = plt.GridSpec(6, 4, hspace=0.3, wspace=0.3)
	ax = fig.add_subplot(grid[:3, :], xticklabels=[])
	y_star_ax = fig.add_subplot(grid[3:5, :], sharex=ax)
	ninst_ax = fig.add_subplot(grid[-1:, :], sharex=ax)


	flierprops = dict(marker='+', markerfacecolor='r', markersize=2,
			  linestyle='none', markeredgecolor='k')
	dx = 0.2
	ndx = 0
	plot_data = dict()
	for dt, dtname in zip(all_query, all_query_name):
		X_qr, y_qr, idx_qr = dt	
		if X_qr.shape[0] != 0:
			estimator.fit(X_train, y_train)

			y_qr_pred = estimator.predict(X_qr)
			pos_x = 1.0 + ndx*dx

			ax, y_star_ax, mean, y_min = show_one_rst(
				y=y_qr, y_pred=y_qr_pred, ax=ax, y_star_ax=y_star_ax, 
				ninst_ax=ninst_ax, pos_x=pos_x, color=color_codes[dtname])
			if dt == "DQ":
				feedback_val = copy.copy(mean)
			plot_data[dtname] = dict()
			plot_data[dtname]["idx_qr"] = idx_qr
			plot_data[dtname]["y_qr"] = y_qr
			plot_data[dtname]["y_qr_pred"] = y_qr_pred
		ndx += 1


	# # update DQ to f then estimate RND
	dtname = "DQ_to_RND"
	X_dq, y_dq, idx_dq = DQ	
	X_rnd, y_rnd, idx_rnd = RND	
	print ("Checking shape train/test:", X_train.shape, y_train.shape, X_dq.shape, y_dq.shape)
	X_train_udt, y_train_udt, _, embedding_model = est_alpha_updated(
		X_train=X_train, y_train=y_train, 
		X_test=X_dq, y_test=y_dq, selected_inds=range(len(y_dq)),
		estimator=copy.copy(estimator)
		)

	if type(embedding_model) is not str:
		X_rnd = embedding_model.transform(X_val=X_rnd, get_min_dist=False)
	estimator.fit(X_train_udt, y_train_udt)
	y_rnd_pred = estimator.predict(X_rnd)
	
	pos_x = 1.0 + 3*dx
	ax, y_star_ax, mean, y_min = show_one_rst(
		y=y_rnd, y_pred=y_rnd_pred, ax=ax, y_star_ax=y_star_ax, 
		ninst_ax=ninst_ax, pos_x=pos_x, color=color_codes[dtname])

	plot_data[dtname] = dict()
	plot_data[dtname]["idx_qr"] = idx_rnd
	plot_data[dtname]["y_qr"] = y_rnd
	plot_data[dtname]["y_qr_pred"] = y_rnd_pred
	makedirs(eval_data_file)
	pickle.dump(plot_data, gfile.GFile(eval_data_file, 'w'))
	print ("Save at:", eval_data_file)

	ax.set_yscale('log')
	ax.set_xlabel(r"Query index", **axis_font) 
	ax.set_ylabel(r"|y_obs - y_pred|", **axis_font)
	y_star_ax.set_ylabel(r"y_os", **axis_font)

	ax.tick_params(axis='both', labelsize=12)
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.tight_layout(pad=1.1)

	makedirs(save_at)
	fig.savefig(save_at, transparent=False)
	release_mem(fig)

	print ("Save at: ", save_at)

	return feedback_val

def map_unlbl_data(FLAGS):
	is_save_query = True
	is_load_estimator = False

	savedir = get_savedir(ith_trial=FLAGS.ith_trial)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index = load_data()

	selected_inds = []
	selected_inds_to_estimator = []
	last_feedback = None

	perform_fig = plt.figure(figsize=(10, 8))
	perform_ax = perform_fig.add_subplot(1, 1, 1)
	for qid in range(1, FLAGS.n_run): 
		queried_idxes = list(range(1, qid))
		# # read load queried data
		# # queried_idxes is None mean all we investigate at initial step
		print ("This query time: ", qid)
		print ("===================")
		if qid != 1:
			# queried files
			queried_files = [savedir + "/query_{0}/query_{0}.csv".format(k) for k in queried_idxes]

			# # get calculated  
			# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
			# print ("queried_files", queried_files)
			all_query = get_queried_data(qids=queried_idxes, queried_files=queried_files, 
				unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
				embedding_model="Not yet")
			dq_X, dq_y, dq_idx = all_query[0]
			os_X, os_y, os_idx = all_query[1]
			rnd_X, rnd_y, rnd_idx = all_query[2]

			assert dq_X.shape[0] == len(dq_y)
			assert os_X.shape[0] == len(os_y)
			assert rnd_X.shape[0] == len(rnd_y)

			# # remove all labeled data of X, y, id to update sampler
			all_id = np.concatenate((dq_idx, os_idx, rnd_idx)).ravel()
			selected_inds = [np.where(unlbl_index==k)[0][0] for k in all_id]

			update_bag = []
			if "DQ" in FLAGS.estimator_update_by:
				update_bag.append(dq_idx)
			if "OS" in FLAGS.estimator_update_by:
				update_bag.append(os_idx)
			if "RND" in FLAGS.estimator_update_by:
				update_bag.append(rnd_idx)
			dt2estimator = np.concatenate(update_bag).ravel()
			selected_inds_to_estimator = [np.where(unlbl_index==k)[0][0] for k in dt2estimator]

		# # 1. load estimator
		if is_load_estimator:
			estimator = load_pickle(est_file)
		elif FLAGS.score_method == "u_gp_mt":	
			if qid == 1:
				update_coeff = 1.0
			else:
				kernel_cfg_file = savedir+"/query_{}".format(qid-1) + "/kernel_cfg.txt"
				update_coeff = np.loadtxt(kernel_cfg_file)	
			mt_kernel=mt_kernel * update_coeff
		else:
			mt_kernel = None

		estimator = utils.get_model(
			FLAGS.score_method, FLAGS.ith_trial, 
			FLAGS.is_search_params, n_shuffle=10000,
			mt_kernel=mt_kernel)


		# # 2. prepare embedding space 
		# # if FLAGS.embedding_method as "org", unlbl_X_sampler is exactly same as unlbl_X 
		_, _, unlbl_X_sampler, _ = est_alpha_updated(
			X_train=X_train, y_train=y_train, 
			X_test=unlbl_X, y_test=unlbl_y, 
			selected_inds=selected_inds, 
			# method=update_all,
			estimator=estimator)


		# # 3. prepare sampler
		uniform_sampler = AL_MAPPING['uniform'](unlbl_X_sampler, unlbl_y, FLAGS.ith_trial)
		sampler = get_AL_sampler(FLAGS.sampling_method)
		sampler = sampler(unlbl_X_sampler, unlbl_y, FLAGS.ith_trial)
		
		est_file = savedir+"/query_{0}/pre_trained_est_{0}.pkl".format(qid)
		
		# # tsne
		"""
		plot current state of hypothetical structures + querying 
		"""
		tsne_file = result_dropbox_dir+"/dim_reduc/"+FLAGS.data_init+".pkl"

		# # to force parameter search
		# estimator.estimator = None
		_x_train, _y_train, estimator, embedding_model = query_and_learn(
			FLAGS=FLAGS, qid=qid,
			selected_inds=selected_inds, 
			selected_inds_to_estimator=selected_inds_to_estimator,
			estimator=estimator,
			X_train=X_train, y_train=y_train, index_train=index_train,
			unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			sampler=sampler, uniform_sampler=uniform_sampler, 
			is_save_query=is_save_query, savedir=savedir, tsne_file=tsne_file,
			is_plot=False, perform_ax=perform_ax, perform_fig=perform_fig)
		makedirs(est_file)
		pickle.dump(estimator, gfile.GFile(est_file, 'w'))

		save_at = savedir+"/query_{}".format(qid)+"/query_performance.pdf"
		eval_data_file = savedir+"/query_{}".format(qid)+"/eval_query_{}.pkl".format(qid) 
		
		"""
		# # It's time to create an error map of samples in each query batch
		"""

		# # 2. put this_queried_files to database for querying results
		this_queried_file = [savedir+"/query_{0}/query_{0}.csv".format(qid)]
		# # get calculated  
		# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
		this_query = get_queried_data(qids=[qid], queried_files=this_queried_file, 
				unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
				embedding_model=embedding_model)
		# if this_dq_X.shape[0] != 0 and this_os_X.shape[0] != 0 and this_rnd_X.shape[0] != 0:
		feedback_val = evaluation_map(FLAGS=FLAGS,
				X_train=_x_train, y_train=_y_train, 
				index_train=index_train, 
				all_query=this_query, sampler=sampler, 
				uniform_sampler=uniform_sampler,
				save_at=save_at, eval_data_file=eval_data_file,
				estimator=copy.copy(estimator))

		# # create distance matrix
		assert unlbl_X_sampler.shape[0] == unlbl_X.shape[0]
		_unlbl_dist = pairwise.euclidean_distances(unlbl_X_sampler)
		metric_df = pd.DataFrame(_unlbl_dist, index=unlbl_index, columns=unlbl_index)

		save_mkl = savedir+"/query_{0}/{1}_dist.png".format(qid, FLAGS.embedding_method)
		metric_df.to_csv(save_mkl.replace(".png", ".csv"))
		
		plot_heatmap(matrix=metric_df.values, 
				vmin=None, vmax=None, save_file=save_mkl, 
				cmap="jet", title=save_mkl.replace(ALdir, ""),
				lines=None)

		# # interpolation
		if FLAGS.embedding_method != "org_space":
			try:
				scatter_plot_2(x=_x_train[:, 0], y=_x_train[:, 1], 
					z_values=_y_train,
					color_array=_y_train, xvline=None, yhline=None, 
					sigma=None, mode='scatter', lbl=None, name=None, 
					x_label='x', y_label='y', 
					save_file=save_mkl.replace("_dist", "interpolation"), 
					title=save_mkl.replace(ALdir, ""),
					interpolate=False, color='blue', 
					preset_ax=None, linestyle='-.', marker='o')

			except:
				pass
			

		# # create invese_trans
		invs_trans = InversableEmbeddingSpace(invs_emb_method="umap")
		save_invs = savedir+"/query_{0}/umap.png".format(qid)
		invs_trans.fit(_x_train, _y_train, save_invs)


		if FLAGS.score_method == "u_gp_mt":
			if last_feedback is not None:
				# update_coeff = last_feedback / float(feedback_val)
				update_coeff = fix_update_coeff
	 
			last_feedback = copy.copy(feedback_val)
			tmp = savedir+"/query_{}".format(qid) + "/kernel_cfg.txt"
			np.savetxt(tmp, [update_coeff])
		# break



if __name__ == "__main__":

	FLAGS(sys.argv)
	is_param_test = False
	is_spark_run = True

	if is_spark_run:
		pr_file = sys.argv[-1]
		kwargs = load_pickle(filename=pr_file)
		FLAGS.score_method = kwargs["score_method"]
		FLAGS.sampling_method =	kwargs["sampling_method"]
		FLAGS.embedding_method = kwargs["embedding_method"]
		FLAGS.active_p = kwargs["active_p"]
		FLAGS.ith_trial = kwargs["ith_trial"]

		map_unlbl_data(FLAGS=FLAGS)

	# # test only
	if is_param_test:
		sampling_methods = [
			"uniform", "exploitation", "margin", "expected_improvement"]
		score_methods = ["u_gp", "u_knn", "e_krr"
				# "fully_connected", "ml-gp", "ml-knn"
			]
		embedding_methods = ["org_space", "MLKR", "LFDA", "LMNN"]
		all_kwargs = list(product(sampling_methods, score_methods, embedding_methods))

		for kw in all_kwargs:
			sampling_method, score_method, embedding_method = kw[0], kw[1], kw[2]
			FLAGS.score_method = score_method
			FLAGS.sampling_method = sampling_method
			FLAGS.embedding_method = embedding_method
			FLAGS.n_run = 2 # # work only with test params

			print ("score_method", FLAGS.score_method)
			print ("sampling_method", FLAGS.sampling_method)
			print ("embedding_method", FLAGS.embedding_method)

			# rank_unlbl_data() # 014 for u_gp
			map_unlbl_data(FLAGS=FLAGS)

