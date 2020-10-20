

import sys, pickle, functools, json, copy
import numpy as np 
from params import *
from absl import app
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
from utils.utils import load_pickle
from utils.general_lib import get_basename, merge_two_dicts
from proc_results import read_exp_params, params2text

from utils import utils
from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from utils.manifold_processing import Preprocessing
from utils.plot import *
from tensorflow.io import gfile

from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvas
import cv2 as cv
import random, re
from scipy.interpolate import griddata
from deformation import read_deformation
from query2vasp import get_qrindex
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


def load_unlbl_data(unlbl_job, result_file):
	# # round data
	# round1 = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/with_standard_ene/mix/rand1___ofm1_no_d.csv"
	# round1_df = pd.read_csv(round1, index_col=0)

	# # read load unlbl data
	unlbl_file = ALdir+"/data/SmFe12/unlabeled_data/"+unlbl_job 
	data = load_pickle(unlbl_file+".pkl")
	unlbl_X = data["data"]
	unlbl_y = data["target"]
	unlbl_index = data["index"]

	unlbl_dir = result_file.replace(".pkl","")+"/"+unlbl_job
	return unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir


def id_qr_to_database(id_qr, db_results, crs_db_results=None, fine_db_results=None):
	# id_qr = arg[0]
	feature_dir = localdir + "/input/feature/"
	assert feature_dir in id_qr
	id_qr_cvt = id_qr.replace(feature_dir, "")
	assert "ofm1_no_d/" in id_qr_cvt
	assert ".ofm1_no_d" in id_qr_cvt

	id_qr_cvt = id_qr_cvt.replace("ofm1_no_d/", "").replace(".ofm1_no_d", "")
	id_qr_cvt = id_qr_cvt.replace("/", '-_-')

	if id_qr_cvt in db_results.index:
		target_y = db_results.loc[id_qr_cvt, "energy_substance_pa"]
	elif id_qr_cvt in fine_db_results.index:
		target_y = fine_db_results.loc[id_qr_cvt, "energy_substance_pa"]
		print ("Add fine_relax results", target_y)
	elif id_qr_cvt in crs_db_results.index:
		target_y = crs_db_results.loc[id_qr_cvt, "energy_substance_pa"]
		print ("Add coarse_relax results", target_y)
	else:
		target_y = None
		print ("None index:", id_qr_cvt, len(db_results.index))
	# if np.isnan(target_y):
	# 	target_y = None
	return (id_qr, id_qr_cvt, target_y)


def get_queried_data(queried_files, database_results, unlbl_X, unlbl_index,
			coarse_db_rst, fine_db_rst):
	"""
	database_results: *.csv of all vasp calculated data, normally in the standard step
	queried_files: all queried files
	unlbl_X: original ublbl data, before any querying
	unlbl_index: original index of ublbl data, before any querying
	"""

	dqs = []
	oss = []
	rnds = []

	frames = [pd.read_csv(k, index_col=0) for k in database_results]
	db_results = pd.concat(frames)
	index_reduce = [get_basename(k) for k in db_results.index]
	db_results["index_reduce"] = index_reduce
	db_results.set_index('index_reduce', inplace=True)

	# # coarse, fine db
	crs_frames = [pd.read_csv(k, index_col=0) for k in coarse_db_rst]
	crs_db_results = pd.concat(crs_frames)
	crs_db_results = crs_db_results.dropna()
	index_reduce = [get_basename(k) for k in crs_db_results.index]
	crs_db_results["index_reduce"] = index_reduce
	crs_db_results.set_index('index_reduce', inplace=True)

	fine_frames = [pd.read_csv(k, index_col=0) for k in fine_db_rst]
	fine_db_results = pd.concat(fine_frames)
	fine_db_results = fine_db_results.dropna()
	index_reduce = [get_basename(k) for k in fine_db_results.index]
	fine_db_results["index_reduce"] = index_reduce
	fine_db_results.set_index('index_reduce', inplace=True)

	for qf in queried_files:
		this_df = pd.read_csv(qf, index_col=0)
		dq, os, rnd = get_qrindex(df=this_df)
		assert len(dq) == 10
		assert len(os) == 10
		assert len(rnd) == 10

		dq_cvt = map(functools.partial(id_qr_to_database, db_results=db_results,
			crs_db_results=crs_db_results, fine_db_results=fine_db_results), dq)
		os_cvt = map(functools.partial(id_qr_to_database, db_results=db_results,
			crs_db_results=crs_db_results, fine_db_results=fine_db_results), os)
		rnd_cvt = map(functools.partial(id_qr_to_database, db_results=db_results,
			crs_db_results=crs_db_results, fine_db_results=fine_db_results), rnd)

		# dqs = merge_two_dicts(dqs, dq_cvt)
		# dq_cvt = list(map(id_qr_to_database, arg1))
		# os_cvt = list(map(id_qr_to_database, arg2))
		# rnd_cvt = list(map(id_qr_to_database, arg3))
		# queried_data.append((dq_cvt, os_cvt, rnd_cvt))
		dqs.extend(dq_cvt)
		oss.extend(os_cvt)
		rnds.extend(rnd_cvt)

	DQs, OSs, RNDs = np.array(dqs), np.array(oss), np.array(rnds)

	valid_Xyid = []
	fig, ax=plt.subplots(figsize=(12, 12))

	for data in [DQs, OSs, RNDs]:
		_y =  data[:, -1] 
		# # last column as predicted variable
		# # data in the format of: [id_qr, id_qr_cvt, target]
		valid_id = [i for i, val in enumerate(_y) if val != None] 
		_y = np.array([float(k) for k in _y[valid_id]]) 
		_idx = np.array(data[valid_id, 0])

		_X = np.array(unlbl_X[[np.where(unlbl_index==k)[0][0] for k in _idx], :])
		valid_Xyid.append((_X, _y, _idx))
	
	return valid_Xyid


def est_alpha_updated(X_train, y_train, X_test, y_test, selected_inds):
	if selected_inds is not None:
		_x_train = np.concatenate((X_train, X_test[selected_inds]), axis=0)
		_y_train = np.concatenate((y_train, y_test[selected_inds]), axis=0)
		assert X_test[selected_inds].all() != None
		return _x_train, _y_train
	else:
		return X_train, y_train



def plot_and_query(FLAGS, all_results,
	selected_inds, estimator,
	X_trval_csv, y_trval_csv, index_trval_csv, 
	unlbl_file, unlbl_X, unlbl_y, unlbl_index,
	sampler, uniform_sampler, is_save_query, csv_save_dir, tsne_file):
	active_p = 1.0
	batch_size = 10
	batch_outstand = 10
	batch_rand = 10
	plt_mode = "2D" # 3D, 2D, 3D_patch
	# is_load_pre_trained = False

	selected_inds_copy = copy.copy(selected_inds)
	# if is_load_pre_trained:
	try:
		X_all_trans = load_pickle(tsne_file)
		print ("Success in reading")

	except Exception as e:
		X_all = np.concatenate((X_trval_csv, unlbl_X))
		X_all_trans = process_dimensional_reduction(unlbl_X=X_all)
		makedirs(tsne_file)
		pickle.dump(X_all_trans, gfile.GFile(tsne_file, 'w'))

	# x_trval = X_all_trans[:n_trval, 0]
	# y_trval = X_all_trans[:n_trval, 1]

	# x = X_all_trans[n_trval:, 0]
	# y = X_all_trans[n_trval:, 1]

	# # x, y to plot
	x = X_all_trans[:, 0]
	y = X_all_trans[:, 1]

	scaler = MinMaxScaler()
	x = scaler.fit_transform(x.reshape(-1, 1)).ravel()
	y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		# # save querying data 
		query_data = dict()
		query_data["unlbl_file"] = unlbl_file
		query_data["unlbl_index"] = unlbl_index
		# # end save querying data 

		m, c = exp_params["m"], exp_params["c"]
		csv_saveat = csv_save_dir + "/m{0}_c{1}.csv".format(m, c)

		accuracies = np.array(result_dict["accuracy"])
		acc_cv_train = np.array(result_dict["cv_train_model"])

		models = [k.estimator.get_params() for k in result_dict["save_model"]]
		GSCVs = [k.GridSearchCV.best_score_ for k in result_dict["save_model"]]

		# shfl_indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise, idx_train, idx_val, idx_test = result_dict["all_X"]

		# estimator = result_dict["save_model"][-1] # # ".estimator" return GaussianRegressor, otherwise return estimator used in sampler
		# estimator = utils.get_model(FLAGS.score_method, FLAGS.seed, False) # FLAGS.is_search_params
		# estimator.fit(X_trval_csv, y_trval_csv)

		_x_train, _y_train = est_alpha_updated(
			X_train=X_trval_csv, y_train=y_trval_csv, 
			X_test=unlbl_X, y_test=unlbl_y, 
			selected_inds=selected_inds_copy)
		
		estimator.fit(_x_train, _y_train)

		unlbl_y_pred = estimator.predict(unlbl_X)
		query_data["unlbl_y_pred"] = unlbl_y_pred

		select_batch_inputs = {"model": estimator, "labeled": None, 
				"eval_acc": None, "X_test": None,	"y_test": None, "y": None, "verbose": True,
				"y_star": min(_y_train)}
		N_unlbl = unlbl_X.shape[0]
		n_sample = min(batch_size, N_unlbl)
			

		# # 1. update by D_{Q}
		new_batch, acq_val = select_batch(sampler, uniform_sampler, active_p, n_sample,
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
		print ("CHECK selected_inds_copy:", len(unlbl_y_pred), len(selected_inds_copy))
		assert outstand_idx != []
		outstand_list = outstand_idx[:batch_outstand]
		lim_outstand_list = max(unlbl_y_pred[outstand_list])

		if is_save_query:
			query_outstanding = np.array([None] * len(unlbl_index))
			query_outstanding[outstand_list] = "query_outstanding"
			query_data["query_outstanding"] = query_outstanding

		selected_inds_copy.extend(outstand_list)
		max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

		# # 3. select by D_{rand}
		the_rest = list(set(range(N_unlbl)) - set(selected_inds_copy))
		random_list = random.sample(the_rest, batch_rand)

		if is_save_query:
			query_random = np.array([None] * len(unlbl_index))
			query_random[random_list] = "query_random"
			query_data["query_random"] = query_random
			print ("query2update_DQ", len(query_data["query2update_DQ"]))
			print ("acq_val", len(query_data["acq_val"]))
			print ("query_outstanding", len(query_data["query_outstanding"]))
			print ("query_random", len(query_data["query_random"]))

			query_df = pd.DataFrame().from_dict(query_data)
			makedirs(csv_saveat)
			query_df.to_csv(csv_saveat)
			print("Save query data at:", csv_saveat)	

		selected_inds_copy.extend(random_list)

		# # AL points ~ smallest min margin ~ biggest apparent points
		if FLAGS.sampling_method == "margin":
			acq_val[np.isinf(acq_val)] = np.max(acq_val)
		scaler = MinMaxScaler()
		size_points = scaler.fit_transform(acq_val.reshape(-1, 1))

		# # name, color, marker for plot
		plot_index = np.concatenate((unlbl_index, index_trval_csv), axis=0)
		name = [k.replace(unlbl_file, "") for k in plot_index]
		family = [get_family(k) for k in plot_index]

		list_cdict = np.array([get_color_112(k) for k in name])
		marker_array = np.array([get_marker_112(k) for k in family])
		alphas = np.array([0.3] * len(plot_index))
		alphas[selected_inds_copy] = 1.0 
		alphas[len(unlbl_index):] = 1.0


		lim_acq_val = min(acq_val[new_batch])

		z =  np.concatenate((unlbl_y_pred, y_trval_csv)) # unlbl_y_pred, min_margin
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
		
		# # acp_val map
		lim_acq_val = min(acq_val[new_batch])
		z =  np.concatenate((acq_val, [0]*len(y_trval_csv))) # unlbl_y_pred, min_margin
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

		break
	return _x_train, _y_train, estimator


def evaluation_map(FLAGS, all_results,
	X_train, y_train, index_trval_csv, 
	all_query,
	sampler, uniform_sampler, save_at, eval_data_file,
	estimator):
	"""
	# # to create an error map of samples in each query batch
	"""
	DQ, OS, RND = all_query 
	all_query_name = ["DQ", "OS", "RND"]
	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		m, c = exp_params["m"], exp_params["c"]
		# csv_saveat = csv_save_dir + "/m{0}_c{1}.csv".format(m, c)

		accuracies = np.array(result_dict["accuracy"])
		acc_cv_train = np.array(result_dict["cv_train_model"])

		models = [k.estimator.get_params() for k in result_dict["save_model"]]
		GSCVs = [k.GridSearchCV.best_score_ for k in result_dict["save_model"]]

		# estimator = result_dict["save_model"][-1] # # ".estimator" return GaussianRegressor, otherwise return estimator used in sampler
		# estimator = utils.get_model(FLAGS.score_method, FLAGS.seed, False) # FLAGS.is_search_params
		
		# fig, ax =plt.subplots(figsize=(8, 8))
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
				print ("X_qr.shpae", X_qr.shape)		
				estimator.fit(X_train, y_train)

				y_qr_pred = estimator.predict(X_qr)
				pos_x = 1.0 + ndx*dx

				ax, y_star_ax, mean, y_min = show_one_rst(
					y=y_qr, y_pred=y_qr_pred, ax=ax, y_star_ax=y_star_ax, 
					ninst_ax=ninst_ax, pos_x=pos_x, color=color_codes[dtname])

				plot_data[dtname] = dict()
				plot_data[dtname]["idx_qr"] = idx_qr
				plot_data[dtname]["y_qr"] = y_qr
				plot_data[dtname]["y_qr_pred"] = y_qr_pred
			
				print ("=============")
			ndx += 1
		
		# # update DQ to f then estimate RND
		dtname = "DQ_to_RND"
		X_dq, y_dq, idx_dq = DQ	
		X_rnd, y_rnd, idx_rnd = RND	

		X_train_udt, y_train_udt = est_alpha_updated(
			X_train=X_train, y_train=y_train, 
			X_test=X_dq, y_test=y_dq, selected_inds=range(len(y_dq)))
		estimator.fit(X_train_udt, y_train_udt)
		y_rnd_pred = estimator.predict(X_rnd)
		
		pos_x = 1.0 + 3*dx
		ax, y_star_ax, mean, y_min = show_one_rst(
			y=y_rnd, y_pred=y_rnd_pred, ax=ax, y_star_ax=y_star_ax, 
			ninst_ax=ninst_ax, pos_x=pos_x, color=color_codes[dtname])

		plot_data[dtname] = dict()
		plot_data[dtname]["idx_qr"] = idx_qr
		plot_data[dtname]["y_qr"] = y_qr
		plot_data[dtname]["y_qr_pred"] = y_qr_pred

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

		makedirs(eval_data_file)
		pickle.dump(plot_data, gfile.GFile(eval_data_file, 'w'))
		print ("Save at:", eval_data_file)
		break	


def map_unlbl_data(ith_trial):
	current_tsne_map = True
	is_save_query = True
	is_load_estimator = False

	unlbl_job = "mix" # mix, "mix_2-24"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	all_results = load_pickle(result_file)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	n_trval = len(X_trval_csv)
	
	# deformations = read_deformation(qr_indexes=index_trval_csv)
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(
		unlbl_job=unlbl_job,  result_file=result_file)

	selected_inds = []
	database_jobs = [
		"mix/query_1.csv", 	"mix/supp_2.csv", "mix/supp_3.csv", "mix/supp_4.csv",  
		"mix/supp_5.csv", "mix/supp_6.csv", "mix/supp_7.csv", "mix/supp_8.csv",
						# "mix_2-24/query_1.csv"
						]
	database_results = [database_dir+"/"+k for k in database_jobs]
	fine_db_rst = [fine_db_dir+"/"+k for k in database_jobs]
	coarse_db_rst = [coarse_db_dir+"/"+k for k in database_jobs]

	for next_query_idx in range(17, 51): 
		if next_query_idx == 1:
			curr_lbl_num_id = None
		queried_idxes = range(1, next_query_idx)
		# # read load queried data
		# # queried_idxes is None mean all we investigate at initial step
		if next_query_idx != 1:
		# if True:
			# qr_dir = csv_save_dir.replace(result_dropbox_dir, __ALdir__+"input/origin_struct/queries/")
			# # vasp run results

			# queried files
			queried_files = [unlbl_dir + "/query_{}".format(k) + "/m0.1_c0.1.csv" for k in queried_idxes]
			# queried_files = [unlbl_dir + "/query_{}".format(next_query_idx) + "/m0.1_c0.1.csv"]

			# # get calculated  
			# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
			# print ("queried_files", queried_files)
			valid_Xyid = get_queried_data(queried_files=queried_files, database_results=database_results, 
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

			# unlbl_index = np.delete(unlbl_index, curr_lbl_num_id)
			# unlbl_X = np.delete(unlbl_X, curr_lbl_num_id, axis=0)
			# unlbl_y = np.delete(unlbl_y, curr_lbl_num_id, axis=0)
			unlbl_y[selected_inds] = all_unlbl_y

			# print ("unlbl_X shape in: ", next_query_idx, "queried: ", unlbl_X.shape)
			# print ("selected_inds: ", next_query_idx, "queried: ", len(selected_inds))

		# # prepare sampler
		uniform_sampler = AL_MAPPING["uniform"](unlbl_X, unlbl_y, FLAGS.seed)
		sampler = get_AL_sampler(FLAGS.sampling_method)
		sampler = sampler(unlbl_X, unlbl_y, FLAGS.seed)
		
		ith_query_storage = unlbl_dir+"/query_{}".format(next_query_idx)
		est_file = ith_query_storage + "/pre_trained_est.pkl"

		if is_load_estimator:
			estimator = load_pickle(est_file)
		else:
			estimator = utils.get_model(
				FLAGS.score_method, FLAGS.seed, 
				FLAGS.is_search_params, n_shuffle=10000)
			makedirs(est_file)
			pickle.dump(estimator, gfile.GFile(est_file, 'w'))
		# # tsne
		if current_tsne_map:
			"""
			plot current state of hypothetical structures + querying 
			"""
			tsne_file = result_dropbox_dir+"/dim_reduc/"+unlbl_job+".pkl"

			print ("Intended save at:", ith_query_storage)
			_x_train, _y_train, estimator = plot_and_query(FLAGS=FLAGS, all_results=all_results,
				selected_inds=selected_inds, estimator=estimator,
				X_trval_csv=X_trval_csv, y_trval_csv=y_trval_csv, 
				index_trval_csv=index_trval_csv,
				unlbl_file=unlbl_file, unlbl_X=unlbl_X, unlbl_y=unlbl_y, unlbl_index=unlbl_index,
				sampler=sampler, uniform_sampler=uniform_sampler, 
				is_save_query=is_save_query, csv_save_dir=ith_query_storage, tsne_file=tsne_file)
			makedirs(est_file)
			pickle.dump(estimator, gfile.GFile(est_file, 'w'))

			save_at = unlbl_dir+"/query_{}".format(next_query_idx)+"/query_performance.pdf"
			eval_data_file = ith_query_storage+"/eval_query_{}.pkl".format(next_query_idx) 
		
		"""
		# # It's time to create an error map of samples in each query batch
		"""
		# # 1. update training data by the selected_inds

		print ("X before shape:", X_trval_csv.shape)
		# _x_train, _y_train = est_alpha_updated(
		# 	X_train=X_trval_csv, y_train=y_trval_csv, 
		# 	X_test=unlbl_X, y_test=unlbl_y, 
		# 	selected_inds=selected_inds)

		# print ("X update shape:", _x_train.shape)
		# print ("Total selected_inds:", len(selected_inds))

		# # 2. put this_queried_files to database for querying results
		this_queried_files = [unlbl_dir+"/query_{}".format(next_query_idx)+"/m0.1_c0.1.csv"]
		# # get calculated  
		# # DQs, OSs, RNDs: [0, 1, 2] "original index", "reduce index", "calculated target"
		valid_Xyid = get_queried_data(queried_files=this_queried_files, 
			database_results=database_results, 
			unlbl_X=unlbl_X, unlbl_index=unlbl_index,
			coarse_db_rst=coarse_db_rst, fine_db_rst=fine_db_rst)
		# # alocate representation, label and id of labeled data

		this_dq_X, this_dq_y, this_dq_idx = valid_Xyid[0]
		this_os_X, this_os_y, this_os_idx = valid_Xyid[1]
		this_rnd_X, this_rnd_y, this_rnd_idx = valid_Xyid[2]
		this_query = [[this_dq_X, this_dq_y, this_dq_idx], 
					[this_os_X, this_os_y, this_os_idx], 
					[this_rnd_X, this_rnd_y, this_rnd_idx]]


		# if this_dq_X.shape[0] != 0 and this_os_X.shape[0] != 0 and this_rnd_X.shape[0] != 0:
		evaluation_map(FLAGS=FLAGS, all_results=all_results,
				X_train=_x_train, y_train=_y_train, 
				index_trval_csv=index_trval_csv, 
				all_query=this_query, sampler=sampler, 
				uniform_sampler=uniform_sampler,
				save_at=save_at, eval_data_file=eval_data_file,
				estimator=estimator)
		# break

if __name__ == "__main__":
	FLAGS(sys.argv)

	# rank_unlbl_data(ith_trial="000") # 014 for u_gp

	map_unlbl_data(ith_trial="000")

