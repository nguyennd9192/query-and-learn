

import sys, pickle, functools
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

__localdir__ = "/Volumes/Nguyen_6TB/work/SmFe12_screening"
__ALdir__ = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master"


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

def rank_unlbl_data(ith_trial):
	active_p = 1.0
	batch_size = 10
	batch_outstand = 10
	batch_rand = 10
	unlbl_job = "mix" # mix, mix_2-24

	result_dir = get_savedir()
	filename = get_savefile()

	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	all_results = load_pickle(result_file)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()

	# # # round data
	# round1 = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/with_standard_ene/mix/rand1___ofm1_no_d.csv"
	# round1_df = pd.read_csv(round1, index_col=0)

	# # read load unlbl data
	
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, N_unlbl, uniform_sampler, sampler, saveat, csv_save_dir = load_unlbl_data(unlbl_job=unlbl_job)
	# # save video at
	height, width = 800, 1200

	out = cv.VideoWriter(saveat, cv.VideoWriter_fourcc(*'MP4V'),30.0,(width,height))

	init_axis = False

	X_trans = process_dimensional_reduction(unlbl_X)
	
	x = X_trans[:, 0]
	y = X_trans[:, 1]	

	# x = unlbl_X[:, 0]
	# y = unlbl_X[:, 1]
	
	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		m, c = exp_params["m"], exp_params["c"]
		accuracies = np.array(result_dict["accuracy"])
		acc_cv_train = np.array(result_dict["cv_train_model"])

		models = [k.estimator.get_params() for k in result_dict["save_model"]]
		GSCVs = [k.GridSearchCV.best_score_ for k in result_dict["save_model"]]

		shfl_indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise, idx_train, idx_val, idx_test = result_dict["all_X"]

		estimator = result_dict["save_model"][-1] # # ".estimator" return GaussianRegressor, otherwise return estimator used in sampler
		print(dir(estimator))
		# estimator = utils.get_model(FLAGS.score_method, FLAGS.seed, False) # FLAGS.is_search_params
		estimator.fit(X_trval_csv, y_trval_csv)

		unlbl_y_pred = estimator.predict(unlbl_X)
		select_batch_inputs = {"model": estimator, "labeled": None, 
				"eval_acc": None, "X_test": None,	"y_test": None, "y": None, "verbose": True,
				"y_star":min(y_trval_csv)}
		n_sample = min(batch_size, N_unlbl)

		# while(len(selected_inds) <= N_unlbl)
		selected_inds = []
		# # 1. update by D_{Q}
		new_batch, acq_val = select_batch(sampler, uniform_sampler, active_p, n_sample,
															 list(selected_inds), **select_batch_inputs)
		selected_inds.extend(new_batch)

		# # 2. select by D_{o/s}
		argsort_y_pred = np.argsort(unlbl_y_pred)
		outstand_idx = [k for k in argsort_y_pred if k not in selected_inds]
		outstand_list = outstand_idx[:batch_outstand]
		max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

		selected_inds.extend(outstand_list)

		# # 3. select by D_{rand}
		the_rest = list(set(range(N_unlbl)) - set(selected_inds))
		random_list = random.sample(the_rest, batch_rand)
		selected_inds.extend(random_list)

		print(unlbl_index[new_batch])
		print(acq_val[new_batch])
		print(min(acq_val), max(acq_val))


		# # AL points ~ smallest min margin ~ biggest apparent points
		acq_val[np.isinf(acq_val)] = np.max(acq_val)
		scaler = MinMaxScaler()
		size_points = scaler.fit_transform(acq_val.reshape(-1, 1))
		# size_points *= 100
		# size_points = 130 - size_points
		# print (min(size_points), max(size_points))

		# # name, color, marker for plot
		name = [k.replace(unlbl_file, "") for k in unlbl_index]
		family = []
		for idx in unlbl_index:
			if "Sm-Fe9" in idx:
				family.append("1-9-3")
			elif "Sm-Fe10" in idx:
				family.append("1-10-2")

		color_array = np.array([get_color_112(k) for k in name])
		marker_array = np.array([get_marker_112(k) for k in family])
		alphas = np.array([0.5] * len(acq_val))
		alphas[selected_inds] = 1.0 

		fig = plt.figure(figsize=(width/100, height/100)) 
		gs = gridspec.GridSpec(nrows=2,ncols=3,figure=fig, width_ratios=[1, 1, 1]
			) 
		canvas = FigureCanvas(fig)
		scatter_plot_3(x=x, y=y, xvlines=None, yhlines=None, s=size_points, alpha=0.2, 
				sigma=None, mode='scatter',	name=None,  
				x_label='Dim 1', y_label="Dim 2", save_file=saveat.replace(".mp4", ".pdf"), 
				interpolate=False, color_array=color_array, 
				preset_ax=None, linestyle='-.', marker=marker_array)

		lim_margin = np.min(acq_val[new_batch])

		# # tSNE map
		ax1 = plt.subplot(gs[0, 0])
		ax_scatter(ax=ax1, x=x, y=y, marker=marker_array, color=color_array,
			 x_label="tSNE axis 1", y_label="tSNE axis 2",
			 alphas=alphas)
		ax_setting()

		# # recommended points 
		ax2 = plt.subplot(gs[0, 1])
		# ax_scatter(ax=ax4,x=min_margin[new_batch],y=np.arange(len(new_batch)),
		# 	marker=marker_array[new_batch],color=color_array[new_batch],
		# 	name=unlbl_index[new_batch])
		ax2.barh(new_batch, acq_val[new_batch], color="black")
		names = [get_basename(k).replace(".ofm1_no_d", "") for k in unlbl_index[new_batch]]
		ax2.set_xlabel("Variance (exploration)", **axis_font)
		ax2.set_ylabel("Selected structures", **axis_font)
		for ith, index in enumerate(new_batch):
			ax2.text(0.2, index+1, names[ith])
		ax_setting()

		# # histogram of acq_val
		ax3 = plt.subplot(gs[1,0])
		plot_hist(x=acq_val, x_label=sampler.name, y_label="Probability density", 
			save_at=None, label=None, nbins=50, ax=ax3)
		ax3.axvline(x=lim_margin, #ymin=0.0, ymax=0.25, # np.max(unlbl_y_pred) 
			label="variance threshold", color="red", linestyle="-.")
		ax3.legend()

		# # acq_val 
		ax4 = plt.subplot(gs[1,1])
		pos = np.arange(len(sampler.name))
		print (unlbl_y_pred)
		ax_scatter(ax=ax4,x=acq_val ,y=unlbl_y_pred,
			marker=marker_array,color=color_array,
			x_label=sampler.name, y_label="Predicted value (exploitation)",
			alphas=alphas)
		# ax4.set_xlim((min(acq_val), max(acq_val)))
		# # get min of criteria to select new structures
		ax4.axvline(x=lim_margin, #ymin=0.0, ymax=0.25, # np.max(unlbl_y_pred) 
			label="variance threshold", color="red", linestyle="-.")
		ax4.axhline(y=max_y_pred_selected, #ymin=0.0, ymax=0.25, # np.max(unlbl_y_pred) 
			label="o/s threshold", color="black", linestyle="-.")
		ax4.legend()
		# print("min pred: ", min(unlbl_y_pred), "max pred: ", max(unlbl_y_pred))
	
		ax5 = plt.subplot(gs[1,2])
		ax_scatter(ax=ax5, x=acq_val ,y=-unlbl_y_pred,
			marker=marker_array,color=color_array,
			x_label=sampler.name, y_label=r"\frac{-ypred}{\sigma}",
			alphas=alphas)
		ax5.axvline(x=lim_margin, #ymin=0.0, ymax=0.25, # np.max(unlbl_y_pred) 
			label="variance threshold", color="red", linestyle="-.")
		# ax5.set_xlim((min(sampler.name), max(sampler.name)))
		ax5.legend()

		# # add text config
		text = params2text(exp_params=exp_params, result_key_to_text=result_key_to_text)
		side_text = plt.figtext(0.75, 0.60, text, bbox=dict(facecolor='white'))
		fig.tight_layout()

		canvas.draw()
		rgba_render = np.array(canvas.renderer.buffer_rgba())
		final_frame = np.delete(rgba_render.reshape(-1,4),3,1)
		# print("shape before:", final_frame.shape)
		final_frame = final_frame.reshape(final_frame.shape[0],final_frame.shape[1],-1)
		final_frame = final_frame.reshape(height, width,-1)
		# print("shape after:", final_frame.shape)
		out.write(final_frame)
		# # save for last frame
		plt.savefig(saveat.replace(".mp4", "_full.pdf"))
		plt.close()

		break
			# if frameNo == 15:
			# 	break
			# break
			# if True:
			# 	ax5 = plt.subplot(gs[0,2])
	out.release()
	cv.destroyAllWindows()
	print("Save at:", saveat)
	# scatter_plot_4(x=x, y=y, 
	# 		# xvlines=[xlb, xub], yhlines=[ylb, yub], 
	# 		xvlines=None, yhlines=None, 
	# 		s=80, alpha=0.2, 
	# 		# title=title,
	# 		sigma=None, mode='scatter', 
	# 		name=None,  # all_local_idxes
	# 		x_label='Dim 1', y_label="Dim 2", 
	# 		save_file=saveat.replace(".pdf", "mix.pdf"),
	# 		interpolate=False, color_array=color_array, 
	# 		preset_ax=None, linestyle='-.', marker=marker_array)


	return unlbl_X, unlbl_index

def load_unlbl_data(unlbl_job, result_file, next_query_idx, queried_idxes):
	# # round data
	# round1 = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/with_standard_ene/mix/rand1___ofm1_no_d.csv"
	# round1_df = pd.read_csv(round1, index_col=0)

	# # read load unlbl data
	unlbl_file = __ALdir__+"/data/SmFe12/unlabeled_data/"+unlbl_job 
	data = load_pickle(unlbl_file+".pkl")
	unlbl_X = data["data"]
	unlbl_y = data["target"]
	unlbl_index = data["index"]

	# # save dir. result_file: saved AL results
	saveat = result_file.replace(".pkl","")+"/"+unlbl_job +"/"+"/cmap_unlbl_rank_unlbl_y_pred.pdf"
	csv_save_dir = result_file.replace(".pkl","")+"/"+unlbl_job+"/query_{}".format(next_query_idx)

	return unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, saveat, csv_save_dir

def id_qr_to_database(id_qr, db_results):
	# id_qr = arg[0]
	feature_dir = __localdir__ + "/input/feature/"
	id_qr_cvt = id_qr.replace(feature_dir, "")
	id_qr_cvt = id_qr_cvt.replace("ofm1_no_d/", "").replace(".ofm1_no_d", "")
	id_qr_cvt = id_qr_cvt.replace("/", '-_-')

	if id_qr_cvt in db_results.index:
		target_y = db_results.loc[id_qr_cvt, "energy_substance_pa"]
	else:
		target_y = None
	print ([id_qr_cvt, target_y])
	return (id_qr, id_qr_cvt, target_y)


def get_queried_data(queried_files, database_results):
	# dqs = dict({})
	# oss = dict({})
	# rnds = dict({})
	dqs = []
	oss = []
	rnds = []

	frames = [pd.read_csv(k, index_col=0) for k in database_results]
	db_results = pd.concat(frames)
	index_reduce = [get_basename(k) for k in db_results.index]
	db_results["index_reduce"] = index_reduce
	db_results.set_index('index_reduce', inplace=True)


	for qf in queried_files:
		this_df = pd.read_csv(qf, index_col=0)
		dq, os, rnd = get_qrindex(df=this_df)

		dq_cvt = map(functools.partial(id_qr_to_database, db_results=db_results), dq)
		os_cvt = map(functools.partial(id_qr_to_database, db_results=db_results), os)
		rnd_cvt = map(functools.partial(id_qr_to_database, db_results=db_results), rnd)

		# dqs = merge_two_dicts(dqs, dq_cvt)
		# dq_cvt = list(map(id_qr_to_database, arg1))
		# os_cvt = list(map(id_qr_to_database, arg2))
		# rnd_cvt = list(map(id_qr_to_database, arg3))
		# queried_data.append((dq_cvt, os_cvt, rnd_cvt))
		dqs.extend(dq_cvt)
		oss.extend(os_cvt)
		rnds.extend(rnd_cvt)

	return np.array(dqs), oss, rnds

def plot_and_query(FLAGS, all_results,
	X_trval_csv, y_trval_csv, index_trval_csv, unlbl_X, unlbl_index,
	sampler, uniform_sampler, active_p, is_save_query):

	tsne_file = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/X_trval_tsne_fit.pkl"
	is_load_pre_trained = False
	if is_load_pre_trained:
		X_all_trans = load_pickle(tsne_file)
	else:
		X_all = np.concatenate((X_trval_csv, unlbl_X))
		X_all_trans = process_dimensional_reduction(unlbl_X=X_all)
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

		shfl_indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise, idx_train, idx_val, idx_test = result_dict["all_X"]

		estimator = result_dict["save_model"][-1] # # ".estimator" return GaussianRegressor, otherwise return estimator used in sampler
		print(dir(estimator))
		# estimator = utils.get_model(FLAGS.score_method, FLAGS.seed, False) # FLAGS.is_search_params
		estimator.fit(X_trval_csv, y_trval_csv)

		unlbl_y_pred = estimator.predict(unlbl_X)
		query_data["unlbl_y_pred"] = unlbl_y_pred

		select_batch_inputs = {"model": estimator, "labeled": None, 
				"eval_acc": None, "X_test": None,	"y_test": None, "y": None, "verbose": True,
				"y_star": min(y_trval_csv)}
		n_sample = min(batch_size, unlbl_X.shape[0])

		# while(len(selected_inds) <= N_unlbl)
		selected_inds = []
		# # 1. update by D_{Q}
		new_batch, acq_val = select_batch(sampler, uniform_sampler, active_p, n_sample,
											list(selected_inds), **select_batch_inputs)
		selected_inds.extend(new_batch)
		if is_save_query:
			query2update_DQ = np.array([None] * len(unlbl_index))
			query2update_DQ[new_batch] = "query2update_DQ"
			query_data["query2update_DQ"] = query2update_DQ
			query_data["variance"] = acq_val


		# # 2. select by D_{o/s}
		argsort_y_pred = np.argsort(unlbl_y_pred)
		outstand_idx = [k for k in argsort_y_pred if k not in selected_inds]
		outstand_list = outstand_idx[:batch_outstand]
		lim_outstand_list = max(unlbl_y_pred[outstand_list])

		if is_save_nquery:
			query_outstanding = np.array([None] * len(unlbl_index))
			query_outstanding[outstand_list] = "query_outstanding"
			query_data["query_outstanding"] = query_outstanding

		selected_inds.extend(outstand_list)
		max_y_pred_selected = np.max(unlbl_y_pred[outstand_list])

		# # 3. select by D_{rand}
		the_rest = list(set(range(N_unlbl)) - set(selected_inds))
		random_list = random.sample(the_rest, batch_rand)

		if is_save_query:
			query_random = np.array([None] * len(unlbl_index))
			query_random[random_list] = "query_random"
			query_data["query_random"] = query_random
			query_df = pd.DataFrame().from_dict(query_data)
			makedirs(csv_saveat)
			query_df.to_csv(csv_saveat)
			print("Save query data at:", csv_saveat)

		selected_inds.extend(random_list)

		# # AL points ~ smallest min margin ~ biggest apparent points
		if FLAGS.sampling_method == "margin":
			acq_val[np.isinf(acq_val)] = np.max(acq_val)
		scaler = MinMaxScaler()
		size_points = scaler.fit_transform(acq_val.reshape(-1, 1))

		# # name, color, marker for plot
		plot_index = np.concatenate((unlbl_index, index_trval_csv), axis=0)
		name = [k.replace(unlbl_file, "") for k in plot_index]
		family = [get_family(k) for k in plot_index]

		color_array = np.array([get_color_112(k) for k in name])
		marker_array = np.array([get_marker_112(k) for k in family])
		alphas = np.array([0.1] * len(plot_index))
		alphas[selected_inds] = 1.0 
		alphas[len(unlbl_index):] = 1.0

		fig = plt.figure(figsize=(10, 8)) 
		ax = fig.add_subplot(1, 1, 1)

		lim_acq_val = min(acq_val[new_batch])

		z =  np.concatenate((unlbl_y_pred, y_trval_csv)) # unlbl_y_pred, min_margin
		xi = np.arange(min(x), max(x), (max(x) - min(x))/200)
		yi = np.arange(min(y), max(y), (max(y) - min(x))/200)
		xi, yi = np.meshgrid(xi,yi)
		# interpolate
		zi = griddata((x,y),z,(xi,yi),method='nearest')
		cs = ax.contourf(xi,yi,zi, levels=20, cmap="jet")
		cbar = fig.colorbar(cs, label="unlbl_y_pred")
		
		# # tSNE map
		ax_scatter(ax=ax, x=x, y=y, marker=marker_array, color=color_array,
			 x_label="tSNE axis 1", y_label="tSNE axis 2",
			 alphas=alphas)
		fig.patch.set_visible(False)
		ax.axis('off')
		plt.tight_layout(pad=1.1)
		plt.savefig(saveat)
		plt.close()


		# # acp_val_map
		fig = plt.figure(figsize=(10, 8)) 
		ax = fig.add_subplot(1, 1, 1)

		lim_acq_val = min(acq_val[new_batch])

		z =  np.concatenate((acq_val, [0]*len(y_trval_csv))) # unlbl_y_pred, min_margin
		xi = np.arange(min(x), max(x), (max(x) - min(x))/200)
		yi = np.arange(min(y), max(y), (max(y) - min(x))/200)
		xi, yi = np.meshgrid(xi,yi)
		# interpolate
		zi = griddata((x,y),z,(xi,yi),method='nearest')
		cs = ax.contourf(xi,yi,zi, levels=20, cmap="jet")
		cbar = fig.colorbar(cs, label="acq_val")
		
		# # tSNE map
		ax_scatter(ax=ax, x=x, y=y, marker=marker_array, color=color_array,
			 x_label="tSNE axis 1", y_label="tSNE axis 2",
			 alphas=alphas)
		fig.patch.set_visible(False)
		ax.axis('off')
		plt.tight_layout(pad=1.1)
		plt.savefig(saveat.replace("unlbl_y_pred", "acq_val"))
		plt.close()

		# # plot traininig validation points only
		#  for idx in unlbl_index:
		# 	if "Sm-Fe9" in idx:
		# 		family.append("1-9-3")
		# 	elif "Sm-Fe10" in idx:
		# 		family.append("1-10-2")
		# color_trval = np.array([get_color_112(k) for k in index_trval_csv])
		# marker_trval = np.array([get_marker_112(k) for k in family])

		# ax_scatter(ax=ax, x=x_trval, y=x_trval, marker=marker_trval, color=color_trval,
		# 	 x_label="tSNE axis 1", y_label="tSNE axis 2",
		# 	 alphas=alphas)
		# index_trval_csv

		scatter_plot_4(x=acq_val, y=unlbl_y_pred, color_array=color_array, 
			xvlines=[lim_acq_val], yhlines=[lim_outstand_list], 
				sigma=None, mode='scatter', lbl=None, name=None, 
				s=80, alphas=alphas, title=None,
				x_label=sampler.name, y_label='unlbl_y_pred', 
				save_file=saveat.replace(".pdf", "_2.pdf"),
				interpolate=False, color='blue', 
				preset_ax=None, linestyle='-.', marker=marker_array)
		# ax_setting()
		# plt.colorbar()
		# plt.title()
		# for spine in plt.gca().spines.values():
		# 	spine.set_visible(False)
		# ax.tick_params(top='off', bottom='off', left='off', right='off', 
		# 		labelleft='off', labelbottom='off')



		break

def map_unlbl_data(ith_trial):
	active_p = 1.0
	batch_size = 10
	batch_outstand = 10
	batch_rand = 10
	queried_idxes = [1] # # update query time
	next_query_idx = max(queried_idxes) + 1
	unlbl_job = "mix" # mix, mix_2-24
	database_dir = __localdir__ + "/result/standard/"
	result_dropbox_dir = __ALdir__ + "/results"

	result_dir = get_savedir()
	filename = get_savefile()
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	all_results = load_pickle(result_file)

	# # get_data_from_flags: get original data obtained from 1st time sampling, not counting other yet.
	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	n_trval = len(X_trval_csv)
	
	# deformations = read_deformation(qr_indexes=index_trval_csv)
	unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, saveat, csv_save_dir = load_unlbl_data(
		unlbl_job=unlbl_job,  result_file=result_file, 
		queried_idxes=queried_idxes, next_query_idx=next_query_idx)
	# # read load queried data
	# # queried_idxes is None mean all we investigate at initial step
	if queried_idxes is not None:
		# qr_dir = csv_save_dir.replace(result_dropbox_dir, __ALdir__+"input/origin_struct/queries/")
		qr_dir = csv_save_dir.replace("query_{}".format(next_query_idx), "")
		queried_files = []
		for k in queried_idxes:
			newdir = qr_dir + "query_{}".format(k) + "/m0.1_c0.1.csv"
			queried_files.append(newdir)
		database_results = [database_dir+"query_{}.csv".format(k) for k in queried_idxes]
		# queried_files = [database_dir+"query_{}_csv".format(k) for k in queried_idxes]
		DQs, OSs, RNDs = get_queried_data(queried_files=queried_files, database_results=database_results)
		print ("Here", DQs[:, 0])
		# print ("DBS:", database_results)

		for tmp in DQs[:, 0]:
			print (tmp in unlbl_index)


	N_unlbl = unlbl_X.shape[0]

	# # prepare sampler
	uniform_sampler = AL_MAPPING["uniform"](unlbl_X, unlbl_y, FLAGS.seed)
	sampler = get_AL_sampler(FLAGS.sampling_method)
	sampler = sampler(unlbl_X, unlbl_y, FLAGS.seed)


	init_axis = False
	is_save_query = False
	# # tsne

	# plot_and_query(FLAGS, all_results,
	# 	X_trval_csv, y_trval_csv, index_trval_csv, unlbl_X, unlbl_index,
	# 	sampler, uniform_sampler, active_p, is_save_query)



if __name__ == "__main__":
	FLAGS(sys.argv)

	# rank_unlbl_data(ith_trial="000") # 014 for u_gp

	map_unlbl_data(ith_trial="000")






