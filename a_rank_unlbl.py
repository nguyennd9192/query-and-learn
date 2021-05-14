

import sys, pickle, functools, json, copy, random, re, time
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

from itertools import product, combinations
from embedding_space import InversableEmbeddingSpace


from sklearn.inspection import partial_dependence, plot_partial_dependence
import multiprocessing
from functools import partial
import matplotlib.cm as cm

from kl import KLdivergence, jensen_shannon_distance, js
from scipy.spatial import distance

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
		is_save_query, savedir, tsne_file, is_plot, perform_ax, perform_fig, pv):
	# is_load_pre_trained = False
	csv_saveat = savedir+"/query_{0}/query_{0}.csv".format(qid)	
	fig_saveat = savedir + "/autism/error_dist.pdf"
	makedirs(csv_saveat)
	makedirs(fig_saveat)

	selected_inds_copy = copy.copy(selected_inds)
	n_unlbl_org = len(unlbl_index)
	n_train_org = X_train.shape[0]
	

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
	fit_estimator = copy.copy(estimator)

	unlbl_y_pred = estimator.predict(_unlbl_X)
	query_data["unlbl_y_pred_{}".format(qid)] = unlbl_y_pred

	select_batch_inputs = {"model": copy.copy(estimator), "labeled": None, 
			"eval_acc": None, "X_test": None, 
			"y_test": None, "y": None, "verbose": True,
			"y_star": min(_y_train),
			"embedding_model":embedding_model,
			"X_org":unlbl_X}

	# # 1. update by D_{Q}
	new_batch, acq_val = select_batch(sampler=sampler, uniform_sampler=uniform_sampler, 
						mixture=FLAGS.active_p, N=FLAGS.batch_size,
						already_selected=list(selected_inds_copy), **select_batch_inputs)
	selected_inds_copy.extend(new_batch)
	if is_save_query:
		query2update_DQ = np.array([None] * n_unlbl_org)
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
		query_outstanding = np.array([None] * n_unlbl_org)
		query_outstanding[outstand_list] = "query_outstanding_{}".format(qid)
		query_data["query_outstanding_{}".format(qid)] = query_outstanding

	selected_inds_copy.extend(outstand_list)
	max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

	# # 3. select by D_{rand}
	random.seed(FLAGS.ith_trial*5*qid + time.time())
	the_non_qr = list(set(range(n_unlbl_org)) - set(selected_inds_copy))
	random_list = random.sample(the_non_qr, FLAGS.batch_rand)
	selected_inds_copy.extend(random_list)

	if is_save_query:
		query_random = np.array([None] * n_unlbl_org)
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
	non_qr_ids = list(set(range(n_unlbl_org)) - set(selected_inds_to_estimator))
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

	if FLAGS.do_plot:
		# # name, color, marker for plot
		plot_index = np.concatenate((unlbl_index, index_train), axis=0)
		family = [get_family(k) for k in plot_index]

		list_cdict = np.array([get_color_112(k) for k in plot_index])
		marker_array = np.array([get_marker_112(k) for k in family])
		alphas = np.array([0.3] * len(plot_index))
		alphas[selected_inds_copy] = 1.0 
		alphas[len(unlbl_index):] = 1.0



		# # plot MLKR space or mds with original space
		if FLAGS.embedding_method != "org_space":
			# # concatenate data points train test
			xy = np.concatenate((_unlbl_X, _x_train[:n_train_org]), axis=0)
		else:
			X_tmp = np.concatenate((unlbl_X, X_train))
			xy = process_dimensional_reduction(X_tmp, method="mds")
			xy *= 10000
		# # selected array as +
		ytrain_pred = estimator.predict(_x_train)

		y_all_pred = np.concatenate((unlbl_y_pred, ytrain_pred[:n_train_org]), axis=0)
		y_all_obs = np.concatenate((unlbl_y, _y_train[:n_train_org]), axis=0)
		error_all = y_all_pred - y_all_obs

		# # merge var all
		var_train = estimator.predict_proba(_x_train)
		var_all = np.concatenate((var, var_train[:n_train_org]), axis=0)  
		this_fig_dir = csv_saveat.replace(".csv", "ipl.pdf")

		marker_array[non_qr_ids] = "." 
		# marker_array[selected_inds] = "o"
		marker_array[new_batch] = "D"
		marker_array[outstand_list] = "*"



		data = dict()
		data["x_embedd"] = xy[:, 0]
		data["y_embedd"] = xy[:, 1]
		data["error"] = error_all

		data["index"] = plot_index
		data["y_obs"] = y_all_obs
		data["y_pred"] = y_all_pred
		data["var"] = var_all
		data["marker"] = marker_array
		plt_df = pd.DataFrame().from_dict(data)
		this_df_dir = this_fig_dir.replace(".pdf", "_plot.csv")

		makedirs(this_df_dir)
		plt_df.to_csv(this_df_dir)

		assert len(pv) == unlbl_X.shape[1]

		X_all = np.concatenate((unlbl_X, X_train))
		# fig, ax = plt.subplots(nrows=1,  sharey=True)

		terms = [ 
			"of",
			"s1", "s2",
			"p1"
			"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "f6"
			"d10"
			# "p1-"
			]

		# with multiprocessing.Pool(processes=8) as pool:
		# 	func = partial(plot_ppde, 
		# 		pv=pv, estimator=fit_estimator, X_train=_x_train, y_train=_y_train,	
		# 		X_all=X_all, xy=xy, savedir=savedir)
		# 	pool.map(func, terms)


		# # # toplot pairplot of original
		# X_pairplot = np.concatenate((unlbl_X[selected_inds_to_estimator], X_train))
		# pairplot_df = pd.DataFrame(X_pairplot, columns=pv)
		# y_zz = np.concatenate((_y_train))
		# pairplot_df[FLAGS.tv] = 

		# # # # toplot ppde of embedding with ft
		# for term in terms:
		# 	this_sdir = savedir + "/query_{0}/ft_ppd/".format(qid)
		# 	plot_ppde(pv=pv, estimator=copy.copy(fit_estimator), 
		# 		X_train=_x_train, y_train=_y_train,
		# 		X_all=X_all, xy=xy, savedir=this_sdir, term=term)

		# 	this_sdir = savedir + "/query_{0}/pairplot/".format(qid)
		# 	pairplot(df=pairplot_df, 
		# 		fix_cols=[FLAGS.tv], term=term, save_dir=this_sdir)
			

		# # # toplot ppde of embedding with ft
		# for i, v in enumerate(pv):
		# 	save_file= savedir+"/query_{0}/ft/{1}.pdf".format(qid, v)
		# 	z_values = X_all[:, i]

		# 	if len(set(z_values)) >1:
		# 		try:
		# 			scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
		# 				z_values=z_values,
		# 				list_cdict=list_cdict, 
		# 				xvlines=[0.0], yhlines=[0.0], 
		# 				sigma=None, mode='scatter', lbl=None, name=None, 
		# 				s=60, alphas=alphas, 
		# 				title=save_file.replace(ALdir, ""),
		# 				x_label=FLAGS.embedding_method + "_dim_1",
		# 				y_label=FLAGS.embedding_method + "_dim_2", 
		# 				interpolate=False, cmap="seismic",
		# 				save_file=save_file,
		# 				preset_ax=None, linestyle='-.', marker=marker_array,
		# 				vmin=None, vmax=None
		# 				)
		# 		except:
		# 			pass

		# # # ===========
		#	
		# # # to plot y_obs, y_pred, var, error, no del
		#
		# # # ===========
		if False:
			save_file=this_fig_dir.replace(".pdf", "_yobs.pdf")
			scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
					z_values=y_all_obs,
					list_cdict=list_cdict, 
					xvlines=[0.0], yhlines=[0.0], 
					sigma=None, mode='scatter', lbl=None, name=None, 
					s=60, alphas=alphas, 
					title=save_file.replace(ALdir, ""),
					x_label=FLAGS.embedding_method + "_dim_1",
					y_label=FLAGS.embedding_method + "_dim_2", 
					save_file=save_file,
					interpolate=False, cmap="PiYG",
					preset_ax=None, linestyle='-.', marker=marker_array,
					vmin=vmin_plt["fe"], vmax=vmax_plt["fe"]
					)



			save_file=this_fig_dir.replace(".pdf", "_error.pdf")
			scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
				z_values=error_all,
				list_cdict=list_cdict, 
				xvlines=[0.0], yhlines=[0.0], 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=60, alphas=alphas, 
				title=save_file.replace(ALdir, ""),
				x_label=FLAGS.embedding_method + "_dim_1",
				y_label=FLAGS.embedding_method + "_dim_2", 
				interpolate=False, cmap="seismic",
				save_file=save_file,
				preset_ax=None, linestyle='-.', marker=marker_array,
				vmin=vmin_plt["fe"]*2, vmax=vmax_plt["fe"]*2
				)


			save_file=this_fig_dir.replace(".pdf", "_ypred.pdf")
			scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
				z_values=y_all_pred,
				list_cdict=list_cdict, 
				xvlines=[0.0], yhlines=[0.0], 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=60, alphas=alphas, 
				title=save_file.replace(ALdir, ""),
				x_label=FLAGS.embedding_method + "_dim_1",
				y_label=FLAGS.embedding_method + "_dim_2", 
				save_file=save_file,
				interpolate=False,  cmap="PuOr",
				preset_ax=None, linestyle='-.', marker=marker_array,
				vmin=vmin_plt["fe"]*2, vmax=vmax_plt["fe"]*2
				)

			save_file=this_fig_dir.replace(".pdf", "_yvar.pdf")
			scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
				z_values=var_all,
				list_cdict=list_cdict, 
				xvlines=[0.0], yhlines=[0.0], 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=60, alphas=alphas, 
				title=save_file.replace(ALdir, ""),
				x_label=FLAGS.embedding_method + "_dim_1",
				y_label=FLAGS.embedding_method + "_dim_2", 
				save_file=save_file,
				interpolate=False, cmap="PRGn",
				preset_ax=None, linestyle='-.', marker=marker_array,
				vmin=-0.01, vmax=1.0
				)

		# # # partial dependence plot

		if False:
			fit_estimator = fit_estimator.fit(_x_train, _y_train)
			n_features = _x_train.shape[1]
			midle = int(n_features/2)
			features = range(n_features)

			if FLAGS.embedding_method == "org_space":
				save_file=this_fig_dir.replace(".pdf", "_avg1.pdf")
				fig = plt.figure(figsize=(8, 8), linewidth=1.0)
				plot_partial_dependence(fit_estimator, _x_train, features[:midle],
					   kind='average', line_kw={"color": "red"})
				makedirs(save_file)
				plt.savefig(save_file, transparent=False)
				release_mem(fig=fig)

				save_file=this_fig_dir.replace(".pdf", "_avg2.pdf")
				fig = plt.figure(figsize=(8, 8), linewidth=1.0)
				plot_partial_dependence(fit_estimator, _x_train,  features[midle:],
					   kind='average', line_kw={"color": "red"})
				makedirs(save_file)
				plt.savefig(save_file, transparent=False)
				release_mem(fig=fig)
			else:
				# _, ax = plt.subplots(ncols=3, figsize=(9, 4))
				fig = plt.figure(figsize=(8, 8), linewidth=1.0)
				save_file=this_fig_dir.replace(".pdf", "_partial.pdf")
				plot_partial_dependence(fit_estimator, _x_train, features,
					   kind='both', n_jobs=3, grid_resolution=20,
						# ax=ax,
						)
				makedirs(save_file)
				plt.savefig(save_file, transparent=False)
				release_mem(fig=fig)

		# # # ===========
		# 
		# # # to create map of maps
		pv_combs = list(combinations(pv, 2))
		with multiprocessing.Pool(processes=8) as pool:
			func = partial(get_cell_distance, 
				pv=pv, X_all=X_all, xy=xy)
			all_dist = pool.map(func, pv_combs)

		sum_dist = np.sum(all_dist, axis=-1)
		print (sum_dist)
		
		save_file= savedir+"/query_{0}/ft/dist_of_features.pdf".format(qid)
		plot_heatmap(sum_dist, vmin=None, vmax=None, 
			save_file=save_file, cmap="jet", lines=None, title=None)

		exit()


	return _x_train, _y_train, _unlbl_X, estimator, embedding_model


def get_cell_distance(pv_comb, pv, X_all, xy):
	# df = pd.DataFrame(0, columns=pv, index=pv)
	n_v = len(pv)
	dist = np.zeros((n_v, n_v))
	v1, v2 = pv_comb[0], pv_comb[1]
	i1, i2 = pv.index(v1), pv.index(v2)
	z1 = X_all[:, i1]
	z2 = X_all[:, i2]

	if len(set(z1)) > 1 and len(set(z2)) > 1:
		z1_on_dist = get_2d_interpolate(x=xy[:, 0], y=xy[:, 1], z_values=z1)
		z2_on_dist = get_2d_interpolate(x=xy[:, 0], y=xy[:, 1], z_values=z2)

		# kl_val = KLdivergence(x=z1_on_dist, y=z2_on_dist)
		# kl_val = jensen_shannon_distance(p=z1_on_dist, q=z2_on_dist)
		# kl_val = distance.jensenshannon(z1_on_dist, z2_on_dist)
		kl_val = js(z1_on_dist, z2_on_dist)
		kl_val = np.nanmean(kl_val)

		dist[i1][i2] = kl_val
		dist[i2][i1] = kl_val

	return dist

def evaluation_map(FLAGS, 
	X_train, y_train, index_train, 
	unlbl_X, unlbl_y, unlbl_index,
	all_query, sampler, uniform_sampler, 
	save_at, eval_data_file, estimator):
	"""
	# # to create an error map of samples in each query batch
	"""
	estimator_copy = copy.copy(estimator)
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
	for idx_qr, dtname in zip(all_query, all_query_name):
		ids = [np.where(unlbl_index==k)[0][0] for k in idx_qr]
		X_qr, y_qr = unlbl_X[ids], unlbl_y[ids]

		if dtname == "DQ":
			X_dq, y_dq, idx_dq = X_qr, y_qr, idx_qr
		if dtname == "RND":
			X_rnd, y_rnd, idx_rnd = X_qr, y_qr, idx_qr

		if X_qr.shape[0] != 0:
			estimator_copy.fit(X_train, y_train)

			y_qr_pred = estimator_copy.predict(X_qr)
			pos_x = 1.0 + ndx*dx

			ax, y_star_ax, mean, y_min = show_one_rst(
				y=y_qr, y_pred=y_qr_pred, ax=ax, y_star_ax=y_star_ax, 
				ninst_ax=ninst_ax, pos_x=pos_x, color=color_codes[dtname])
			if dtname == "DQ":
				feedback_val = copy.copy(mean)
			plot_data[dtname] = dict()
			plot_data[dtname]["idx_qr"] = idx_qr
			plot_data[dtname]["y_qr"] = y_qr
			plot_data[dtname]["y_qr_pred"] = y_qr_pred
		ndx += 1


	# # update DQ to f then estimate RND
	dtname = "DQ_to_RND"
	print ("Checking shape train/test:", X_train.shape, y_train.shape, X_dq.shape, y_dq.shape)
	X_train_udt, y_train_udt, _, embedding_model = est_alpha_updated(
		X_train=X_train, y_train=y_train, 
		X_test=X_dq, y_test=y_dq, selected_inds=range(len(y_dq)),
		estimator=copy.copy(estimator)
		)

	if type(embedding_model) is not str:
		X_rnd = embedding_model.transform(X_val=X_rnd, get_min_dist=False)
	estimator_copy.fit(X_train_udt, y_train_udt)
	y_rnd_pred = estimator_copy.predict(X_rnd)
	
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


	return feedback_val

def map_unlbl_data(FLAGS):
	is_save_query = True
	is_load_estimator = False

	savedir = get_savedir(ith_trial=FLAGS.ith_trial)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = load_data()

	selected_inds = []
	selected_inds_to_estimator = []
	last_feedback = None

	perform_fig = plt.figure(figsize=(10, 8))
	perform_ax = perform_fig.add_subplot(1, 1, 1)
	for qid in range(1, FLAGS.n_run): #
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
			dq_idx, os_idx, rnd_idx = get_queried_data(qids=queried_idxes, queried_files=queried_files, 
				unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
				embedding_model="Not yet")

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
		uniform_sampler = AL_MAPPING['uniform'](X=unlbl_X_sampler, y=unlbl_y, seed=FLAGS.ith_trial)
		sampler = get_AL_sampler(FLAGS.sampling_method)
		sampler = sampler(X=unlbl_X_sampler, y=unlbl_y, seed=FLAGS.ith_trial)
		
		est_file = savedir+"/query_{0}/pre_trained_est_{0}.pkl".format(qid)
		
		# # tsne
		"""
		plot current state of hypothetical structures + querying 
		"""
		tsne_file = result_dropbox_dir+"/dim_reduc/"+FLAGS.data_init+".pkl"

		# # to force parameter search
		# estimator.estimator = None
		_x_train, _y_train, _unlbl_X, estimator, embedding_model = query_and_learn(
			FLAGS=FLAGS, qid=qid,
			selected_inds=selected_inds, 
			selected_inds_to_estimator=selected_inds_to_estimator,
			estimator=estimator,
			X_train=X_train, y_train=y_train, index_train=index_train,
			unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			sampler=sampler, uniform_sampler=uniform_sampler, 
			is_save_query=is_save_query, savedir=savedir, tsne_file=tsne_file,
			is_plot=False, perform_ax=perform_ax, perform_fig=perform_fig,
			pv=pv)
		makedirs(est_file)
		pickle.dump(estimator, gfile.GFile(est_file, 'w'))

		# # new 27Feb

		if FLAGS.embedding_method != "org_space" and FLAGS.do_plot:
			A_matrix = embedding_model.learn_metric.components_.T
			A_matrix_save = savedir+"/query_{0}/Amatrix_{0}.png".format(qid)
			max_abs_A = np.abs(A_matrix).max()
			A_df = pd.DataFrame(A_matrix, index=pv)
			
			plot_heatmap(matrix=A_df.values, 
					vmin=-max_abs_A, vmax=max_abs_A, save_file=A_matrix_save, 
					cmap="bwr", title=A_matrix_save.replace(ALdir, ""),
					lines=None)
			# A_df["pv"] = pv
			A_df.to_csv(A_matrix_save.replace(".png", ".csv"))

			n_inst = unlbl_X.shape[0]
			pairs = np.array(list(product(unlbl_X, unlbl_X)))

			score_matrix = embedding_model.learn_metric.score_pairs(pairs).reshape((n_inst, n_inst))
			score_df = pd.DataFrame(score_matrix, index=unlbl_index, columns=unlbl_index)

			save_mkl = savedir+"/query_{0}/score_{1}.png".format(qid, FLAGS.embedding_method)
			score_df.to_csv(save_mkl.replace(".png", ".csv"))
			
			plot_heatmap(matrix=score_df.values, 
					vmin=None, vmax=None, save_file=save_mkl, 
					cmap="jet", title=save_mkl.replace(ALdir, ""),
					lines=None)
		# # end new
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
		
		feedback_val = evaluation_map(FLAGS=FLAGS,
			X_train=_x_train, y_train=_y_train, 
			unlbl_X=_unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			index_train=index_train, 
			all_query=this_query, sampler=sampler, 
			uniform_sampler=uniform_sampler,
			save_at=save_at, eval_data_file=eval_data_file,
			estimator=estimator)

		# # create distance matrix
		assert unlbl_X_sampler.shape[0] == unlbl_X.shape[0]
		_unlbl_dist = pairwise.euclidean_distances(unlbl_X_sampler)
		metric_df = pd.DataFrame(_unlbl_dist, index=unlbl_index, columns=unlbl_index)

		save_mkl = savedir+"/query_{0}/{1}_dist.png".format(qid, FLAGS.embedding_method)
		metric_df.to_csv(save_mkl.replace(".png", ".csv"))
		
		# plot_heatmap(matrix=metric_df.values, 
		# 		vmin=None, vmax=None, save_file=save_mkl, 
		# 		cmap="jet", title=save_mkl.replace(ALdir, ""),
		# 		lines=None)

		# # create invese_trans
		# invs_trans = InversableEmbeddingSpace(invs_emb_method="umap")
		# save_invs = savedir+"/query_{0}/umap.png".format(qid)
		# invs_trans.fit(_x_train, _y_train, save_invs)


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
	else:
		map_unlbl_data(FLAGS=FLAGS)


	
