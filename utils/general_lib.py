
import os, glob, ntpath, pickle, functools
import pandas as pd
import numpy as np
from tensorflow.io import gfile
from params import *

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

def load_pickle(filename):
	if not gfile.exists(filename):
		raise NameError("ERROR: the following data not available \n" + filename)
	data = pickle.load(gfile.GFile(filename, "rb"))
	return data

def get_qrindex(df):
	update_DQ_str = df.loc[df["query2update_DQ"]=="query2update_DQ", "unlbl_index"].to_list()
	outstand_str = df.loc[df["query_outstanding"]=="query_outstanding", "unlbl_index"].to_list()
	random_str = df.loc[df["query_random"]=="query_random", "unlbl_index"].to_list()
	return update_DQ_str, outstand_str, random_str


def est_alpha_updated(X_train, y_train, X_test, y_test, selected_inds):
	if selected_inds is not None:
		_x_train = np.concatenate((X_train, X_test[selected_inds]), axis=0)
		_y_train = np.concatenate((y_train, y_test[selected_inds]), axis=0)
		assert X_test[selected_inds].all() != None
		return _x_train, _y_train
	else:
		return X_train, y_train


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
		# print ("Add fine_relax results", target_y)
	elif id_qr_cvt in crs_db_results.index:
		target_y = crs_db_results.loc[id_qr_cvt, "energy_substance_pa"]
		# print ("Add coarse_relax results", target_y)
	else:
		target_y = None
		# print ("None index:", id_qr_cvt, len(db_results.index))
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
	valid_id = [i for i, val in enumerate(_y) if val != None] 
	_y = np.array([float(k) for k in _y[valid_id]]) 

	# # _idx is index in queried file, unlbl_X
	_idx = np.array(data_cvt[valid_id, 0])

	_X = np.array(unlbl_X[[np.where(unlbl_index==k)[0][0] for k in _idx], :])
	valid_Xyid = (_X, _y, _idx)

	return valid_Xyid
