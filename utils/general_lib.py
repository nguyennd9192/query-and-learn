

import os, glob, ntpath, pickle, functools, copy, sys
from params import *

import pandas as pd
import numpy as np
from tensorflow.io import gfile

try:
	from utils import utils # # ignore in call create_data
except:
	pass

from embedding_space import EmbeddingSpace
from sklearn.preprocessing import MinMaxScaler


vmin_plt = dict({"fe":-0.8, "magmom_pa":1.2})
vmax_plt = dict({"fe":0.2, "magmom_pa":2.2})



def release_mem(fig):
	fig.clf()
	plt.close()
	gc.collect()

def ax_setting():
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)

def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))


def get_subdirs(sdir):
	subdirs = glob.glob(sdir+"/*")
	return subdirs

def get_basename(filename):
		head, tail = ntpath.split(filename)
		basename = os.path.splitext(tail)[0]
		return tail

def merge_two_dicts(x, y):
		"""Given two dictionaries, merge them into a new dict as a shallow copy."""
		z = x.copy()
		z.update(y)
		return z

def filter_array(x, y):
	# for i in y:
	# 	if type(i) != np.float64:
	# 		print (i, type(i))
	nan_x = np.isnan(x)
	nan_y = np.isnan(np.array(y).astype(np.float64))
	nan_id = nan_x + nan_y
	return x[~nan_id], y[~nan_id]


def load_pickle(filename):
	if not gfile.exists(filename):
		raise NameError("ERROR: the following data not available \n" + filename)
	data = pickle.load(gfile.GFile(filename, "rb"))
	return data

def load_flags(filename):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(filename, type=argparse.FileType('r'))
	args = parser.parse_args()

	return args


def dump_pickle(data, filename):
	pickle.dump(data, gfile.GFile(filename, 'wb'))

def dump_flags(data, filename):
	FLAGS.append_flags_into_file(filename)



def get_savedir(ith_trial):
	s_dir = str(os.path.join(
			FLAGS.save_dir,
			"/".join([FLAGS.data_init, FLAGS.sampling_method,
						FLAGS.score_method, FLAGS.embedding_method, 
						FLAGS.mae_update_threshold, str(FLAGS.active_p),
						str(FLAGS.is_search_params)
						])))


	result_file = s_dir + "/trial_{}.pkl".format(ith_trial)
	unlbl_dir = result_file.replace(".pkl","/")

	if FLAGS.score_method == "u_gp_mt":
		mt_kernel = 1.0 # 1.0, 0.001
		fix_update_coeff = 1
		unlbl_dir += "_mt{}".format(mt_kernel)

	# # to mark whether update estimator by DQ only or DQ vs RND
	# # "" is update all
	# estimator_update_by = str(FLAGS.estimator_update_by).split('_') # , "RND", "OS"
	# if len(estimator_update_by) < 3:
	# 	for k in estimator_update_by:
	unlbl_dir += FLAGS.estimator_update_by

	return unlbl_dir


def get_qrindex(df, qid):
	dq = 	"query2update_DQ_{}".format(qid)
	os = 	"query_outstanding_{}".format(qid)
	rnd = 	"query_random_{}".format(qid)



	update_DQ_str = df.loc[df[dq]==dq, "unlbl_index"].to_list()
	outstand_str = df.loc[df[os]==os, "unlbl_index"].to_list()
	random_str = df.loc[df[rnd]==rnd, "unlbl_index"].to_list()
	return update_DQ_str, outstand_str, random_str


def est_alpha_updated(X_train, y_train, 
				X_test, y_test, selected_inds, 
				estimator):
	# # 1. filter 
	# # hold or dimiss
	model = "empty"
	if selected_inds != []:
		# # update X_train, y_train by selected_inds
		if selected_inds is not None:
			X_train = np.concatenate((X_train, X_test[selected_inds]), axis=0)
			y_train = np.concatenate((y_train, y_test[selected_inds]), axis=0)
			assert X_test[selected_inds].all() != None

		# if FLAGS.mae_update_threshold != "update_all":
		# 	estimator.fit(X_train, y_train)
		# 	errors = estimator.predict(X_train)
		# 	for ith, err in zip(errors):
		# 		if err > FLAGS.mae_update_threshold:
		# 			X_train.remove(ith)
		# 			y_train.remove(ith)


	# # normalize
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# # transform to embedding or not
	if FLAGS.embedding_method != "org_space":
		model = EmbeddingSpace(embedding_method=FLAGS.embedding_method)
		if FLAGS.embedding_method == "LMNN":
			_y_train = np.round(y_train, 1)#.reshape(len(y_train), 1) #.astype(str)
			print ("_y_train", _y_train.shape)
			# print ("_y_train", len(set(_y_train)))

		else:
			_y_train = copy.copy(y_train)

		model.fit(X_train=X_train, y_train=_y_train)
		X_train = model.transform(X_val=X_train, get_min_dist=False)
		X_test = model.transform(X_val=X_test, get_min_dist=False)

	return X_train, y_train, X_test, model


def load_data():
	# # read_load train data
	if FLAGS.is_test_separate:
		X_train, y_train, index_train = utils.get_mldata(FLAGS.data_dir, FLAGS.data_init)
		X_test, y_test, index_test = utils.get_mldata(FLAGS.data_dir, FLAGS.data_target)

	return X_train, y_train, index_train, X_test, y_test, index_test

def norm_id(id_qr):
	n_dir = "/Volumes/Nguyen_6TB/work/SmFe12_screening"
	assert n_dir in id_qr
	id_qr_cvt = id_qr.replace(n_dir, "")
	# id_qr_cvt = get_basename(id_qr)
	rmvs = ["/result/", "/input/",  "feature/",
			"coarse_relax/", "fine_relax/", "standard/",
			"ofm1_no_d/", ".ofm1_no_d", "_cnvg",
			"query_1/",  "supp_2/", "supp_3/", "supp_4/",  
	    	"supp_5/", "supp_6/", "mix/supp_7/", "supp_8/",
	    	"supp_9/", "supp_10/",
			"mix/", "mix-_-", "init/"
			]
	for rm in rmvs:
		id_qr_cvt = id_qr_cvt.replace(rm, "")

	id_qr_cvt = id_qr_cvt.replace("/", '-_-')
	return id_qr_cvt

def id_qr_to_database(id_qr, std_results, 
		coarse_results=None, fine_results=None, tv="energy_substance_pa"):
	# # Still use in create_data
	# id_qr = arg[0]
	id_qr_cvt = norm_id(id_qr)
	if id_qr_cvt in std_results.index:
		target_y = std_results.loc[id_qr_cvt, tv]
	elif id_qr_cvt in fine_results.index:
		target_y = fine_results.loc[id_qr_cvt, tv]
		# print ("Add fine_relax results", target_y)
	elif id_qr_cvt in coarse_results.index:
		target_y = coarse_results.loc[id_qr_cvt, tv]
		# print ("Add coarse_relax results", target_y)
	else:
		target_y = None
		# print ("None index:", id_qr_cvt, len(std_results.index))
	return id_qr, id_qr_cvt, target_y


def id_qr(qr, y, index):
	if qr in index:
		idx = np.where(index==qr)[0][0]
		value = y[idx]
	else:
		value = None
	return (qr, value)



def get_queried_data(qids, queried_files, unlbl_X, unlbl_y, unlbl_index,
			 embedding_model):
	"""
	database_results: *.csv of all vasp calculated data, normally in the standard step
	queried_files: all queried files
	unlbl_X: original ublbl data, before any querying
	unlbl_index: original index of ublbl data, before any querying
	"""

	dqs = []
	oss = []
	rnds = []

	for qid, qf in zip(qids, queried_files):
		this_df = pd.read_csv(qf, index_col=0)
		dq, os, rnd = get_qrindex(df=this_df, qid=qid)

		dq_cvt = map(functools.partial(id_qr, y=unlbl_y, index=unlbl_index), dq)
		os_cvt = map(functools.partial(id_qr, y=unlbl_y, index=unlbl_index), os)
		rnd_cvt = map(functools.partial(id_qr, y=unlbl_y, index=unlbl_index), rnd)

		dqs.extend(dq_cvt)
		oss.extend(os_cvt)
		rnds.extend(rnd_cvt)

	DQs, OSs, RNDs = np.array(dqs), np.array(oss), np.array(rnds)

	valid_Xyid = []

	for data in [DQs, OSs, RNDs]:
		_y =  data[:, -1] 
		valid_id = [i for i, val in enumerate(_y) if val != None] 
		_idx = data[valid_id, 0]
		valid_Xyid.append(_idx)
	
	return valid_Xyid


def get_all_unlb_y(database_results, unlbl_X, unlbl_index,
			coarse_db_rst, fine_db_rst):
	"""
	database_results: *.csv of all vasp calculated data, normally in the standard step
	queried_files: all queried files
	unlbl_X: original ublbl data, before any querying
	unlbl_index: original index of ublbl data, before any querying
	"""

	dqs = []

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


	data_cvt = map(functools.partial(id_qr_to_database, db_results=db_results,
		crs_db_results=crs_db_results, fine_db_results=fine_db_results), unlbl_index)
	data_cvt = np.array(list(data_cvt))

	valid_Xyid = []

	_y =  data_cvt[:, -1] 
	# # last column as predicted variable
	# # data in the format of: [id_qr, id_qr_cvt, target]
	valid_id = [i for i, val in enumerate(_y) if val is not None and val is not np.nan] 
	_y = np.array([float(k) for k in _y[valid_id]]) 

	# # _idx is index in queried file, unlbl_X
	_idx = np.array(data_cvt[valid_id, 0])

	_X = np.array(unlbl_X[[np.where(unlbl_index==k)[0][0] for k in _idx], :])
	valid_Xyid = (_X, _y, _idx)

	return valid_Xyid


def vasp_lbl2mix(unlbl_file, database_results, coarse_db_rst, fine_db_rst):
	"""
	database_results: *.csv of all vasp calculated data, normally in the standard step
	queried_files: all queried files
	unlbl_X: original ublbl data, before any querying
	unlbl_index: original index of ublbl data, before any querying
	"""
	unlbl_df = pd.read_csv(unlbl_file, index_col=0)
	unlbl_index = unlbl_df.index.to_list()

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

	y_index_cvt = map(functools.partial(id_qr_to_database, db_results=db_results,
		crs_db_results=crs_db_results, fine_db_results=fine_db_results), unlbl_index)
	y_index_cvt = np.array(list(y_index_cvt))
	unlbl_df["y_obs"] = None
	for a in y_index_cvt:
		id_qr, id_qr_cvt, target_y = a[0], a[1], a[2]
		if target_y != None:
			unlbl_df.loc[id_qr, "y_obs"] = target_y

	unlbl_df.to_csv(unlbl_file.replace(".csv","_with_lbl.csv",))
	



