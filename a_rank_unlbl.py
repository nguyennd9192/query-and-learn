

import sys, pickle, functools, json, copy, random, re
import numpy as np 
from params import *
from absl import app
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
from utils.utils import load_pickle
from utils.general_lib import *

from utils import utils
from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from proc_results import read_exp_params, params2text

from utils.manifold_processing import Preprocessing
from utils.plot import *
from tensorflow.io import gfile

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvas
import cv2 as cv
from scipy.interpolate import griddata
from deformation import read_deformation
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support



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

def process_dimensional_reduction(unlbl_X):
	# # tsne
	is_tsne = False
	is_mds = True
	is_isomap = False # True

	processing = Preprocessing()
	# config_tsne = dict({"n_components":2, "perplexity":500.0,  # same meaning as n_neighbors
	# 	"early_exaggeration":1000.0, # same meaning as n_cluster
	# 	"learning_rate":1000.0, "n_iter":1000,
	# 	 "n_iter_without_progress":300, "min_grad_norm":1e-07, 
	# 	 "metric":'euclidean', "init":'random',
	# 	 "verbose":0, "random_state":None, "method":'barnes_hut', 
	# 	 "angle":0.5, "n_jobs":None})
	# processing.similarity_matrix = unlbl_X
	# X_trans, _, a, b = processing.tsne(**config_tsne)
	if is_mds:
		config_mds = dict({"n_components":2, "metric":True, "n_init":4, "max_iter":300, "verbose":0,
				"eps":0.001, "n_jobs":None, "random_state":None, "dissimilarity":'precomputed'})
		cosine_distance = 1 - cosine_similarity(unlbl_X)
		processing.similarity_matrix = cosine_distance
		X_trans, _ = processing.mds(**config_mds)

	if is_isomap:
		config_isomap = dict({"n_neighbors":5, "n_components":2, "eigen_solver":'auto', "tol":0, 
				"max_iter":None, "path_method":'auto', "neighbors_algorithm":'auto', "n_jobs":None,
				"metric":"l1"})
		processing.similarity_matrix = unlbl_X
		X_trans, a, b = processing.iso_map(**config_isomap)
	return X_trans


def query_and_learn(FLAGS, 
		selected_inds, selected_inds_to_estimator,
		estimator, X_trval, y_trval, index_trval, 
		unlbl_file, unlbl_X, unlbl_y, unlbl_index, sampler, uniform_sampler, 
		is_save_query, csv_save_dir, tsne_file, is_plot):
	active_p = 1.0
	plt_mode = "2D" # 3D, 2D, 3D_patch
	# is_load_pre_trained = False

	selected_inds_copy = copy.copy(selected_inds)
	# if is_load_pre_trained:

	if is_plot:
		try:
			X_all_trans = load_pickle(tsne_file)
			print ("Success in reading")

		except Exception as e:
			X_all = np.concatenate((X_trval, unlbl_X))
			X_all_trans = process_dimensional_reduction(unlbl_X=X_all)
			makedirs(tsne_file)
			pickle.dump(X_all_trans, gfile.GFile(tsne_file, 'w'))

		# # x, y to plot
		x = X_all_trans[:, 0]
		y = X_all_trans[:, 1]

		scaler = MinMaxScaler()
		x = scaler.fit_transform(x.reshape(-1, 1)).ravel()
		y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

	# # save querying data 
	query_data = dict()
	query_data["unlbl_file"] = unlbl_file
	query_data["unlbl_index"] = unlbl_index
	# # end save querying data 

	csv_saveat = csv_save_dir + "/query.csv"

	# # update train, test by selected inds
	_x_train, _y_train, _unlbl_X, embedding_model = est_alpha_updated(
		X_train=X_trval, y_train=y_trval, 
		X_test=unlbl_X, y_test=unlbl_y, 
		selected_inds=selected_inds_to_estimator,
		embedding_method=FLAGS.embedding_method,
		mae_update_threshold=FLAGS.mae_update_threshold,
		estimator=estimator) # # in the past: selected_inds (update by all database)

	# # fit with whole
	estimator.fit(_x_train, _y_train)

	unlbl_y_pred = estimator.predict(_unlbl_X)
	query_data["unlbl_y_pred"] = unlbl_y_pred

	select_batch_inputs = {"model": estimator, "labeled": None, 
			"eval_acc": None, "X_test": None, 
			"y_test": None, "y": None, "verbose": True,
			"y_star": min(_y_train)}

	# # 1. update by D_{Q}
	new_batch, acq_val = select_batch(sampler, uniform_sampler, active_p, FLAGS.batch_size,
						list(selected_inds_copy), **select_batch_inputs)
	selected_inds_copy.extend(new_batch)
	if is_save_query:
		query2update_DQ = np.array([None] * len(unlbl_index))
		query2update_DQ[new_batch] = "query2update_DQ"
		query_data["query2update_DQ"] = query2update_DQ
		query_data["acq_val"] = acq_val


	# # 2. select by D_{o/s}
	argsort_y_pred = np.argsort(unlbl_y_pred)
	outstand_idx = [k for k in argsort_y_pred if k not in selected_inds_copy]
	assert outstand_idx != []
	outstand_list = outstand_idx[:FLAGS.batch_outstand]
	lim_outstand_list = max(unlbl_y_pred[outstand_list])

	if is_save_query:
		query_outstanding = np.array([None] * len(unlbl_index))
		query_outstanding[outstand_list] = "query_outstanding"
		query_data["query_outstanding"] = query_outstanding

	selected_inds_copy.extend(outstand_list)
	max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

	# # 3. select by D_{rand}
	the_rest = list(set(range(_unlbl_X.shape[0])) - set(selected_inds_copy))
	random_list = random.sample(the_rest, FLAGS.batch_rand)

	if is_save_query:
		query_random = np.array([None] * len(unlbl_index))
		query_random[random_list] = "query_random"
		query_data["query_random"] = query_random

		query_df = pd.DataFrame().from_dict(query_data)
		makedirs(csv_saveat)
		query_df.to_csv(csv_saveat)

	selected_inds_copy.extend(random_list)

	# # AL points ~ smallest min margin ~ biggest apparent points
	

	if is_plot:
		if FLAGS.sampling_method == "margin":
			acq_val[np.isinf(acq_val)] = np.max(acq_val)
	
		scaler = MinMaxScaler()
		size_points = scaler.fit_transform(acq_val.reshape(-1, 1))
		# # name, color, marker for plot
		plot_index = np.concatenate((unlbl_index, index_trval), axis=0)
		name = [k.replace(unlbl_file, "") for k in plot_index]
		family = [get_family(k) for k in plot_index]

		list_cdict = np.array([get_color_112(k) for k in name])
		marker_array = np.array([get_marker_112(k) for k in family])
		alphas = np.array([0.3] * len(plot_index))
		alphas[selected_inds_copy] = 1.0 
		alphas[len(unlbl_index):] = 1.0


		lim_acq_val = min(acq_val[new_batch])

		z =  np.concatenate((unlbl_y_pred, y_trval)) # unlbl_y_pred, min_margin
		x1 = np.arange(min(x), max(x), (max(x) - min(x))/200)
		y1 = np.arange(min(y), max(y), (max(y) - min(x))/200)
		xi, yi = np.meshgrid(x1,y1)
		# interpolate
		zi = griddata((x,y),z,(xi,yi),method='nearest')
		if plt_mode == "3D_patch":
			xi, yi, zi = x, y, z
		ax = ax_surf(xi=xi, yi=yi, zi=zi, label="pred_val", mode=plt_mode)

		# # tSNE map
		csv_save_dir += "/"+plt_mode
		save_figat = csv_save_dir+"/cmap_unlbl_rank_unlbl_y_pred.pdf"
		ax_scatter(ax=ax, x=x, y=y, marker=marker_array, list_cdict=list_cdict,
			 x_label="tSNE axis 1", y_label="tSNE axis 2",
			 alphas=alphas, save_at=save_figat, plt_mode=plt_mode)
		
		# # new plot
		ax2 = ax_surf(xi=xi, yi=yi, zi=zi, label="pred_val", mode=plt_mode)
		list_cdict2 = np.array(copy.copy(list_cdict))
		marker_array2 = np.array(copy.copy(marker_array))
		mask = np.full(len(list_cdict2),False,dtype=bool)
		mask[selected_inds_copy] = True # # for selected ids
		mask[-len(index_trval):] = True # # for obs dataset
		list_cdict2[~mask] = dict({"grey":"full"})
		marker_array2[~mask] = "o"

		ax_scatter(ax=ax2, x=x, y=y, marker=marker_array2, 
			list_cdict=list_cdict2,
			x_label="tSNE axis 1", y_label="tSNE axis 2",
			alphas=alphas, plt_mode=plt_mode,
			save_at=save_figat.replace(".pdf", "2.pdf"))
		
		# # acp_val map
		lim_acq_val = min(acq_val[new_batch])
		z =  np.concatenate((acq_val, [0]*len(y_trval))) # unlbl_y_pred, min_margin
		x1 = np.arange(min(x), max(x), (max(x) - min(x))/200)
		y1 = np.arange(min(y), max(y), (max(y) - min(x))/200)
		xi, yi = np.meshgrid(x1,y1)
		
		# interpolate
		zi = griddata((x,y),z,(xi,yi),method='nearest')
		if plt_mode == "3D_patch":
			xi, yi, zi = x, y, z

		ax = ax_surf(xi=xi, yi=yi, zi=zi, 
			label="acp_val", mode=plt_mode)
		# plt.show()

		# # tSNE map
		ax_scatter(ax=ax, x=x, y=y, marker=marker_array, list_cdict=list_cdict,
			 x_label="tSNE axis 1", y_label="tSNE axis 2",
			 alphas=alphas, plt_mode=plt_mode,
			 save_at=save_figat.replace("unlbl_y_pred", "acq_val"))
		try:
			scatter_plot_5(x=acq_val, y=unlbl_y_pred, list_cdict=list_cdict, 
				xvlines=[lim_acq_val], yhlines=[lim_outstand_list], 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=80, alphas=alphas, title=None,
				x_label=sampler.name, y_label='unlbl_y_pred', 
				save_file=save_figat.replace(".pdf", "_2.pdf"),
				interpolate=False, 
				preset_ax=None, linestyle='-.', marker=marker_array)
		except Exception as e:
			pass

	return _x_train, _y_train, estimator, embedding_model


def evaluation_map(FLAGS, X_train, y_train, index_trval, 
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

	X_train_udt, y_train_udt, _, embedding_model = est_alpha_updated(
		X_train=X_train, y_train=y_train, 
		X_test=X_dq, y_test=y_dq, selected_inds=range(len(y_dq)),
		embedding_method=FLAGS.embedding_method,
		mae_update_threshold=FLAGS.mae_update_threshold,
		estimator=estimator
		)

	if embedding_model != "empty":
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
	plt.savefig(save_at, transparent=False)
	print ("Save at: ", save_at)


	return feedback_val



def map_unlbl_data(ith_trial, FLAGS):
	is_save_query = True
	is_load_estimator = False

	unlbl_job = "mix" # mix, "mix_2-24"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	# all_results = load_pickle(result_file)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_trval, y_trval, index_trval, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	n_trval = len(X_trval)
	
	# deformations = read_deformation(qr_indexes=index_trval)
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(
		unlbl_job=unlbl_job,  result_file=result_file)

	if FLAGS.score_method == "u_gp_mt":
		mt_kernel = 1.0 # 1.0, 0.001
		fix_update_coeff = 1
		unlbl_dir += "_mt{}".format(mt_kernel)

	# # to mark whether update estimator by DQ only or DQ vs RND
	# # "" is update all
	estimator_update_by = ["DQ"] # , "RND", "OS"
	if len(estimator_update_by) < 3:
		for k in estimator_update_by:
			unlbl_dir += k

	selected_inds = []
	selected_inds_to_estimator = []
	last_feedback = None

	for next_query_idx in range(1, FLAGS.n_run):  # 51
		if next_query_idx == 1:
			curr_lbl_num_id = None
		queried_idxes = range(1, next_query_idx)
		# # read load queried data
		# # queried_idxes is None mean all we investigate at initial step
		if next_query_idx != 1:
			# queried files
			queried_files = [unlbl_dir + "/query_{}".format(k) + "/query.csv" for k in queried_idxes]
			# queried_files = [unlbl_dir + "/query_{}".format(next_query_idx) + "/m0.1_c0.1.csv"]

			# # get calculated  
			# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
			# print ("queried_files", queried_files)
			valid_Xyid = get_queried_data(queried_files=queried_files, database_results=database_results, 
				unlbl_X=unlbl_X, unlbl_index=unlbl_index,
				coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst,
				embedding_model=embedding_model)
			# # alocate representation, label and id of labeled data
			dq_X, dq_y, dq_idx = valid_Xyid[0]
			os_X, os_y, os_idx = valid_Xyid[1]
			rnd_X, rnd_y, rnd_idx = valid_Xyid[2]

			all_query = [[dq_X, dq_y, dq_idx], [os_X, os_y, os_idx], [rnd_X, rnd_y, rnd_idx]]
			# print ("DQ, OS, RND shape")
			assert dq_X.shape[0] == dq_y.shape[0]
			assert os_X.shape[0] == os_X.shape[0]
			assert rnd_X.shape[0] == rnd_X.shape[0]

			# print ("Querying ith:", next_query_idx, dq_X.shape, dq_y.shape, dq_idx.shape)
			# print ("Querying ith:", next_query_idx, os_X.shape, os_X.shape, os_X.shape)
			# print ("Querying ith:", next_query_idx, rnd_X.shape, rnd_X.shape, rnd_X.shape)
			print  ("This query time: ", next_query_idx)
			print ("===================")

			# # remove all labeled data of X, y, id to update sampler
			all_lbl_id = np.concatenate((dq_idx, os_idx, rnd_idx)).ravel()
			all_unlbl_y = np.concatenate((dq_y, os_y, rnd_y)).ravel()
			all_unlbl_X = np.concatenate((dq_X, os_X, rnd_X), axis=0)

			selected_inds = [np.where(unlbl_index==k)[0][0] for k in all_lbl_id]

			tmp = []
			if "DQ" in estimator_update_by:
				tmp.append(dq_idx)
			if "OS" in estimator_update_by:
				tmp.append(os_idx)
			if "RND" in estimator_update_by:
				tmp.append(rnd_idx)
			dt2estimator = np.concatenate(tmp).ravel()
			selected_inds_to_estimator = [np.where(unlbl_index==k)[0][0] for k in dt2estimator]

			# unlbl_index = np.delete(unlbl_index, curr_lbl_num_id)
			# unlbl_X = np.delete(unlbl_X, curr_lbl_num_id, axis=0)
			# unlbl_y = np.delete(unlbl_y, curr_lbl_num_id, axis=0)
			unlbl_y[selected_inds] = all_unlbl_y

			# print ("unlbl_X shape in: ", next_query_idx, "queried: ", unlbl_X.shape)
			# print ("selected_inds: ", next_query_idx, "queried: ", len(selected_inds))

		# # 1. load previous update info of mt_kernel
		if is_load_estimator:
			estimator = load_pickle(est_file)
		elif FLAGS.score_method == "u_gp_mt":	
			if next_query_idx == 1:
				update_coeff = 1.0
			else:
				kernel_cfg_file = unlbl_dir+"/query_{}".format(next_query_idx-1) + "/kernel_cfg.txt"
				update_coeff = np.loadtxt(kernel_cfg_file)				
			estimator = utils.get_model(
				FLAGS.score_method, FLAGS.seed, 
				FLAGS.is_search_params, n_shuffle=10000,
				mt_kernel=mt_kernel * update_coeff)
		else:
			estimator = utils.get_model(
				FLAGS.score_method, FLAGS.seed, 
				FLAGS.is_search_params, n_shuffle=10000,
				mt_kernel=None)


		# # 2. prepare embedding space 
		# # if FLAGS.embedding_method as "org",
		# # unlbl_X_sampler is exactly same as unlbl_X
		_, _, unlbl_X_sampler, _ = est_alpha_updated(
			X_train=X_trval, y_train=y_trval, 
			X_test=unlbl_X, y_test=unlbl_y, 
			selected_inds=selected_inds,
			embedding_method=FLAGS.embedding_method,
			mae_update_threshold=FLAGS.mae_update_threshold,
			estimator=estimator)


		# # 3. prepare sampler
		uniform_sampler = AL_MAPPING["uniform"](unlbl_X_sampler, unlbl_y, FLAGS.seed)
		sampler = get_AL_sampler(FLAGS.sampling_method)
		sampler = sampler(unlbl_X_sampler, unlbl_y, FLAGS.seed)
		
		ith_query_storage = unlbl_dir+"/query_{}".format(next_query_idx)
		est_file = ith_query_storage + "/pre_trained_est.pkl"
		
		# # tsne
		"""
		plot current state of hypothetical structures + querying 
		"""
		tsne_file = result_dropbox_dir+"/dim_reduc/"+unlbl_job+".pkl"

		# # to force parameter search
		# estimator.estimator = None
		_x_train, _y_train, estimator, embedding_model = query_and_learn(
			FLAGS=FLAGS, 
			selected_inds=selected_inds, 
			selected_inds_to_estimator=selected_inds_to_estimator,
			estimator=estimator,
			X_trval=X_trval, y_trval=y_trval, index_trval=index_trval,
			unlbl_file=unlbl_file, unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
			sampler=sampler, uniform_sampler=uniform_sampler, 
			is_save_query=is_save_query, csv_save_dir=ith_query_storage, tsne_file=tsne_file,
			is_plot=False)
		makedirs(est_file)
		pickle.dump(estimator, gfile.GFile(est_file, 'w'))

		save_at = unlbl_dir+"/query_{}".format(next_query_idx)+"/query_performance.pdf"
		eval_data_file = ith_query_storage+"/eval_query_{}.pkl".format(next_query_idx) 
		
		"""
		# # It's time to create an error map of samples in each query batch
		"""

		# # 2. put this_queried_files to database for querying results
		this_queried_files = [unlbl_dir+"/query_{}".format(next_query_idx)+"/query.csv"]
		# # get calculated  
		# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
		valid_Xyid = get_queried_data(queried_files=this_queried_files, 
			database_results=database_results, 
			unlbl_X=unlbl_X, unlbl_index=unlbl_index,
			coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst,
			embedding_model=embedding_model)
		# # alocate representation, label and id of labeled data

		this_dq_X, this_dq_y, this_dq_idx = valid_Xyid[0]
		this_os_X, this_os_y, this_os_idx = valid_Xyid[1]
		this_rnd_X, this_rnd_y, this_rnd_idx = valid_Xyid[2]
		this_query = [[this_dq_X, this_dq_y, this_dq_idx], 
					[this_os_X, this_os_y, this_os_idx], 
					[this_rnd_X, this_rnd_y, this_rnd_idx]]


		# if this_dq_X.shape[0] != 0 and this_os_X.shape[0] != 0 and this_rnd_X.shape[0] != 0:
		feedback_val = evaluation_map(FLAGS=FLAGS,
				X_train=_x_train, y_train=_y_train, 
				index_trval=index_trval, 
				all_query=this_query, sampler=sampler, 
				uniform_sampler=uniform_sampler,
				save_at=save_at, eval_data_file=eval_data_file,
				estimator=estimator)
		if FLAGS.score_method == "u_gp_mt":
			if last_feedback is not None:
				# update_coeff = last_feedback / float(feedback_val)
				update_coeff = fix_update_coeff
	 
			last_feedback = copy.copy(feedback_val)
			tmp = unlbl_dir+"/query_{}".format(next_query_idx) + "/kernel_cfg.txt"
			np.savetxt(tmp, [update_coeff])
		# break



if __name__ == "__main__":
	FLAGS(sys.argv)

	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	FLAGS.score_method = kwargs["score_method"]
	FLAGS.sampling_method =	kwargs["sampling_method"]
	FLAGS.embedding_method = kwargs["embedding_method"]
 	# #
	print ("FLAGS", FLAGS.score_method)

	# rank_unlbl_data(ith_trial="000") # 014 for u_gp

	map_unlbl_data(ith_trial="000", FLAGS=FLAGS)

