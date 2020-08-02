

import sys
import numpy as np
from params import *
from absl import app
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
from utils.utils import load_pickle
from proc_results import read_exp_params

from utils import utils
from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sklearn.preprocessing import MinMaxScaler

from utils.manifold_processing import Preprocessing
from utils.plot import get_color_112, get_marker_112, scatter_plot_3, scatter_plot_4

def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
									 **kwargs):
		n_active = int(mixture * N)
		n_passive = N - n_active
		kwargs["N"] = n_active
		kwargs["already_selected"] = already_selected
		batch_AL, min_margin = sampler.select_batch(**kwargs)
		already_selected = already_selected + batch_AL
		kwargs["N"] = n_passive
		kwargs["already_selected"] = already_selected
		# kwargs_copy = copy.copy(kwargs)
		# if "return_best_sim" in kwargs_copy.keys():
		#   del kwargs_copy[key]
		# batch_PL = uniform_sampler.select_batch(**kwargs_copy)
		batch_PL, p = uniform_sampler.select_batch(**kwargs)
		return batch_AL + batch_PL, min_margin

def rank_unlbl_data(ith_trial):
	active_p  =1.0
	batch_size = 10

	result_dir = get_savedir()
	filename = get_savefile()

	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	all_results = load_pickle(result_file)
	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	print(X_trval_csv.shape)
	print(y_trval_csv.shape)

	# # read load unlbl data
	unlbl_file = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/unlabeled_data/mix"
	data = load_pickle(unlbl_file+".pkl")
	unlbl_X = data["data"]
	unlbl_y = data["target"]
	unlbl_index = data["index"]

	N_unlbl = unlbl_X.shape[0]

	# # prepare sampler
	uniform_sampler = AL_MAPPING["uniform"](unlbl_X, unlbl_y, FLAGS.seed)

	sampler = get_AL_sampler(FLAGS.sampling_method)
	sampler = sampler(unlbl_X, unlbl_y, FLAGS.seed)

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

		estimator = result_dict["save_model"][-1].estimator


		estimator = utils.get_model(FLAGS.score_method, FLAGS.seed, False) # FLAGS.is_search_params
		estimator.fit(X_trval_csv, y_trval_csv)

		unlbl_y_pred = estimator.predict(unlbl_X)
		select_batch_inputs = {
				"model": estimator, #
				"labeled": None,
				"eval_acc": None,
				"X_test": None,
				"y_test": None,
				"y": None,
				"verbose": True
				}
		n_sample = min(batch_size, N_unlbl)

		selected_inds = []
		new_batch, min_margin = select_batch(sampler, uniform_sampler, active_p, n_sample,
															 list(selected_inds), **select_batch_inputs)
		print (new_batch)

		rank_ind = np.argsort(min_margin)
		print (unlbl_index[rank_ind])

		break

	# # tsne
	config_tsne = dict({"n_components":2, "perplexity":500.0,  # same meaning as n_neighbors
					"early_exaggeration":400.0, # same meaning as n_cluster
					"learning_rate":1000.0, "n_iter":1000,
					 "n_iter_without_progress":300, "min_grad_norm":1e-07, 
					 "metric":'euclidean', "init":'random',
					 "verbose":0, "random_state":None, "method":'barnes_hut', 
					 "angle":0.5, "n_jobs":None})
	processing = Preprocessing()
	processing.similarity_matrix = unlbl_X
	X_trans, _, a, b = processing.tsne(**config_tsne)
	
	name = [k.replace(unlbl_file, "") for k in unlbl_index]
	family = []
	for idx in unlbl_index:
		if "Sm-Fe9" in idx:
			family.append("1-9-3")
		elif "Sm-Fe10" in idx:
			family.append("1-10-2")

	color_array = [get_color_112(k) for k in name]
	marker_array = [get_marker_112(k) for k in family]

	x = X_trans[:, 0]
	y = X_trans[:, 1]

	scaler = MinMaxScaler()
	size_points = scaler.fit_transform(min_margin.reshape(-1, 1))
	size_points *= 100
	saveat = result_file.replace(".pkl","") + "/unlbl_rank.pdf"
	scatter_plot_3(x=x, y=y, 
			# xvlines=[xlb, xub], yhlines=[ylb, yub], 
			xvlines=None, yhlines=None, 
			s=size_points, alpha=0.2, 
			# title=title,
			sigma=None, mode='scatter', 
			name=None,  # all_local_idxes
			x_label='Dim 1', y_label="Dim 2", 
			save_file=saveat,
			interpolate=False, color_array=color_array, 
			preset_ax=None, linestyle='-.', marker=marker_array)

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


	return unlbl_X, unlbl_index, rank_ind


if __name__ == "__main__":
	FLAGS(sys.argv)

	rank_unlbl_data(ith_trial="010")







