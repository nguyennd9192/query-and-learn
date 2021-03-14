from params import *
from utils.plot import *
from general_lib import *

import sys, pickle, functools, json, copy
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np
import pandas as pd

axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
# performance_codes = dict({"uniform":"red", "exploitation":"black", 
# 		"margin":"blue", "expected_improvement":"green"})

performance_codes = dict({"org_space":"blue", "MLKR":"red"}) 
hatch_codes = dict({"uniform":"/", "exploitation":"*", 
				"margin":"o", "expected_improvement":"/"}) 
# # '-', '+', 'x', '\\', '*', 'o', 'O', '.'

def get_embedding_map(qid, all_data):
	# # read load data, save dir, query_file
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = all_data
	savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	
	savedir += "/query_{0}".format(qid)

	query_file = savedir + "/query_{0}.csv".format(qid)

	n_train_org = X_train.shape[0]
	# # read load query data
	if not gfile.exists(query_file):
		print ("This file: ", query_file, "does not hold." )
	df = pd.read_csv(query_file, index_col=0)
	kw = "last_query_{}".format(qid)
	ids_col = df[kw]

	# # selected ids
	selected_inds = np.where(ids_col==kw)[0]

	# # non-selected ids
	ids_non_qr = df[df["last_query_{}".format(qid)].isnull()].index.tolist()
	error_non_qr = df.loc[ids_non_qr, "err_{}".format(qid)]

	_x_train, _y_train, _unlbl_X, embedding_model = est_alpha_updated(
		X_train=X_train, y_train=y_train, 
		X_test=unlbl_X, y_test=unlbl_y, 
		selected_inds=selected_inds,
		estimator=None) 
	list_cdict, marker_array, alphas = get_scatter_config(unlbl_index=unlbl_index, 
		index_train=index_train, selected_inds=selected_inds)
	


	xy = np.concatenate((_unlbl_X, _x_train[:n_train_org]), axis=0)
	y_all_obs = np.concatenate((unlbl_y, _y_train[:n_train_org]), axis=0)
	savedir += "/x_on_embedd".format(qid)	

	X_all = np.concatenate((unlbl_X, X_train))

	for i, v in enumerate(pv):
		print (v)
		save_file = savedir+"/ft/{0}.txt".format(v)
		z_values=X_all[:, i]
		if len(set(z_values)) >1:
			dump_interpolate(x=xy[:, 0], y=xy[:, 1], 
				z_values=z_values,	save_file=save_file,
					)
			save_file = savedir+"/ft_img/{0}.pdf".format(v)
			# scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
			# 		z_values=z_values,
			# 		list_cdict=list_cdict, 
			# 		xvlines=[0.0], yhlines=[0.0], 
			# 		sigma=None, mode='scatter', lbl=None, name=None, 
			# 		s=60, alphas=alphas, 
			# 		title=save_file.replace(ALdir, ""),
			# 		x_label=FLAGS.embedding_method + "_dim_1",
			# 		y_label=FLAGS.embedding_method + "_dim_2", 
			# 		save_file=save_file,
			# 		interpolate=False, cmap="PiYG",
			# 		preset_ax=None, linestyle='-.', marker=marker_array,
			# 		vmin=None, vmax=None
			# 		)

	save_file = savedir + "/y_all_obs.txt"
	dump_interpolate(x=xy[:, 0], y=xy[:, 1], z_values=y_all_obs,
			save_file=save_file)

	save_file = savedir + "/y_all_obs.pdf"
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


def simple_pearson():
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = load_data()
	lim_os = -0.05
	ids = np.where(unlbl_y < lim_os)[0]

	# print (unlbl_y[ids])
	# print (unlbl_index[ids])
	x = np.concatenate((X_train[:, 28], unlbl_X[:, 28]), axis=0)
	y = np.concatenate((y_train, unlbl_y), axis=0)

	scatter_plot(x=x, y=y, xvline=None, yhline=None, 
		sigma=None, mode='scatter', lbl=None, name=None, 
		x_label='x', y_label='y', 
		save_file=result_dropbox_dir+"/test.pdf", interpolate=False, color='blue', 
		linestyle='-.', 
		marker=['o']*len(y), title=None)
	return unlbl_index[ids]


def get_normal(surf):
	grad = np.gradient(surf, edge_order=1)  
	zy, zx = grad[0], grad[1]
	# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
	# to reduce noise
	#zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
	#zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

	normal = np.dstack(grad)
	n = np.linalg.norm(normal, axis=2)
	normal[:, :, 0] /= n
	normal[:, :, 1] /= n
	return normal


	
def two_map_sim(savedir, ref, ft, ref_map):

	savefig = savedir + "/fig"
	ft_file = savedir + "/ft/{}.txt".format(ft)
	ft_map = np.loadtxt(ft_file)

	ref_grad = get_normal(surf=ref_map)
	ft_grad = get_normal(surf=ft_map)
	
	sim_map = np.multiply(ref_grad, ft_grad).sum(2)
	sim_score = np.nanmean(np.abs(sim_map))

	save_at = savedir+"/{0}/{1}.pdf".format(ref, ft)
	two_surf(surf1=ref_map, surf2=ft_map, 
		lbl1=ref, lbl2=ft, title=save_at.replace(ALdir, "") +'\n'+ str(round(sim_score, 3)),
		save_at=save_at)

	# save_file = savedir + "/sim_map.pdf"
	# imshow(grid=sim_map, cmap="tab20c", 
	# 	save_file=save_file, vmin=-0.01, vmax=1)

	# gradient_map(list_vectors=ref_grad, 
	# 	save_file=savefig+"/{}.pdf".format(ref))
	# gradient_map(list_vectors=ft_grad, 
	# 	save_file=savefig +"/{}.pdf".format(ft))
	return sim_score


def map_features(qid):
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = all_data
	savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	
	query_file = savedir + "/query_{0}/query_{0}.csv".format(qid)
	save_file= savedir+"/query_{0}/ft/{1}.txt".format(qid, v)

if __name__ == "__main__":
	FLAGS(sys.argv)
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	FLAGS.score_method = kwargs["score_method"]
	FLAGS.sampling_method =	kwargs["sampling_method"]
	FLAGS.embedding_method = kwargs["embedding_method"]
	FLAGS.active_p = kwargs["active_p"]
	FLAGS.ith_trial = kwargs["ith_trial"]
	# simple_pearson()


	qids = [1] # 5, 10, 15, 20, 25, 30
	all_data = load_data()
	pv = all_data[-1]

	df = pd.DataFrame(index=pv, columns=qids)

	savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	save_score = savedir+"/x_on_embedd_score_qid.csv"
	ref = "y_all_obs"

	for qid in qids:
		get_embedding_map(qid=qid, all_data=all_data)
		qid_dir = savedir + "/query_{}/x_on_embedd".format(qid)

		ref_file = qid_dir + "/{}.txt".format(ref)
		ref_map = np.loadtxt(ref_file)
		for v in pv:
			sim_score = two_map_sim(
				savedir=qid_dir, ref=ref, ft=v,
				ref_map=ref_map)
			df.loc[v, qid] = sim_score
			df.to_csv(save_score)
	print ("save at:", save_score)












		