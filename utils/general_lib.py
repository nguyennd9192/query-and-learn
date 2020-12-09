
import os, glob, ntpath, pickle, functools, copy
import pandas as pd
import numpy as np
from tensorflow.io import gfile
from params import *
from utils.embedding_space import EmbeddingSpace
from sklearn.preprocessing import MinMaxScaler
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
	nan_x = np.isnan(x)
	nan_y = np.isnan(y)
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


def get_qrindex(df):
	update_DQ_str = df.loc[df["query2update_DQ"]=="query2update_DQ", "unlbl_index"].to_list()
	outstand_str = df.loc[df["query_outstanding"]=="query_outstanding", "unlbl_index"].to_list()
	random_str = df.loc[df["query_random"]=="query_random", "unlbl_index"].to_list()
	return update_DQ_str, outstand_str, random_str


def est_alpha_updated(X_train, y_train, 
				X_test, y_test, selected_inds, 
				embedding_method,
				mae_update_threshold, estimator):
	# # 1. filter 
	# # hold or dimiss
	model = "empty"
	if selected_inds != []:
		if mae_update_threshold != "update_all":
			tmp_X = copy.copy(X_test[selected_inds])
			tmp_y = copy.copy(y_test[selected_inds])
			mae = estimator.best_score_(X=tmp_X, y=tmp_y)
			if mae > float(mae_update_threshold):
				selected_inds = None

		# # update X_train, y_train by selected_inds
		if selected_inds is not None:
			X_train = np.concatenate((X_train, X_test[selected_inds]), axis=0)
			y_train = np.concatenate((y_train, y_test[selected_inds]), axis=0)
			assert X_test[selected_inds].all() != None

	# # transform to embedding or not
	if embedding_method != "org_space":
		model = EmbeddingSpace(embedding_method=embedding_method)

		if embedding_method == "LMNN":
			_y_train = np.round(y_train, 1)
		else:
			_y_train = copy.copy(y_train)

		model.fit(X_train=X_train, y_train=_y_train)
		X_train = model.transform(X_val=X_train, get_min_dist=False)
		X_test = model.transform(X_val=X_test, get_min_dist=False)

	# # normalize
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	return X_train, y_train, X_test, model


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
	feature_dir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input/feature/"
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
		# print ("Add fine_relax results", target_y)
	elif id_qr_cvt in crs_db_results.index:
		target_y = crs_db_results.loc[id_qr_cvt, "energy_substance_pa"]
		# print ("Add coarse_relax results", target_y)
	else:
		target_y = None
		# print ("None index:", id_qr_cvt, len(db_results.index))
	return (id_qr, id_qr_cvt, target_y)



def get_queried_data(queried_files, database_results, unlbl_X, unlbl_index,
			coarse_db_rst, fine_db_rst, embedding_model):
	"""
	database_results: *.csv of all vasp calculated data, normally in the standard step
	queried_files: all queried files
	unlbl_X: original ublbl data, before any querying
	unlbl_index: original index of ublbl data, before any querying
	"""

	dqs = []
	oss = []
	rnds = []

	for qf in queried_files:
		this_df = pd.read_csv(qf, index_col=0)
		dq, os, rnd = get_qrindex(df=this_df)

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

	for data in [DQs, OSs, RNDs]:
		_y =  data[:, -1] 
		# # last column as predicted variable
		# # data in the format of: [id_qr, id_qr_cvt, target]
		valid_id = [i for i, val in enumerate(_y) if val != None] 
		_y = np.array([float(k) for k in _y[valid_id]]) 
		_idx = np.array(data[valid_id, 0])

		_X = np.array(unlbl_X[[np.where(unlbl_index==k)[0][0] for k in _idx], :])
		if embedding_model != "org_space":
			_X = embedding_model.transform(X_val=_X, get_min_dist=False)
		valid_Xyid.append((_X, _y, _idx))
	
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
	



