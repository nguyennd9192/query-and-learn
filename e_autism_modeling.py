from params import *
from utils.plot import *
from general_lib import *

import sys, pickle, functools, json, copy
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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

	cmaps = [
			# 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
   #          'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
   #          'hot', 'afmhot', 'gist_heat', 'copper',
   #          'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
   #          'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
   #          'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
   #          'Dark2', 'Set1', 'Set2', 'Set3',
   #          'tab10', 'tab20', 
			'tab20b', 
			# 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
			# 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
			# 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
			# 'gist_ncar'
			]
	for cmap in cmaps:
		for i, v in enumerate(pv):
			if "p1" in v:
				save_file = savedir+"/ft/{0}.txt".format(v)
				z_values = X_all[:, i]
				# print (z_values)
				if len(set(z_values)) >1:
					# dump_interpolate(x=xy[:, 0], y=xy[:, 1], 
					# 	z_values=z_values,	save_file=save_file,
							# )
					save_file = savedir+"/ft_img_cbar/{0}/{1}.pdf".format(cmap, v)
					scatter_plot_7(x=xy[:, 0], y=xy[:, 1], 
							z_values=z_values,
							list_cdict=list_cdict, 
							xvlines=[0.0], yhlines=[0.0], 
							sigma=None, mode='scatter', lbl=None, name=None, 
							s=60, alphas=alphas, 
							title=save_file.replace(ALdir, ""),
							x_label=FLAGS.embedding_method + "_dim_1",
							y_label=FLAGS.embedding_method + "_dim_2", 
							save_file=save_file,
							interpolate=False, cmap=cmap,
							preset_ax=None, linestyle='-.', marker=marker_array,
							vmin=None, vmax=None
							)
			

	# save_file = savedir + "/y_all_obs.txt"
	# dump_interpolate(x=xy[:, 0], y=xy[:, 1], z_values=y_all_obs,
	# 		save_file=save_file)

	# save_file = savedir + "/y_all_obs.pdf"
	# scatter_plot_7(x=xy[:, 0], y=xy[:, 1], 
	# 		z_values=y_all_obs,
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
	# 		vmin=vmin_plt["fe"], vmax=vmax_plt["fe"]
	# 		)


def ofm_diff_map(qid, all_data):

	ofm_diff_file = "{}/data/SmFe12/ofm_diff/summary.csv".format(ALdir)
	ofm_df = pd.read_csv(ofm_diff_file, index_col=0)

	savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	savedir += "/query_{0}".format(qid)
	plot_file = savedir + "/query_{0}ipl_plot.csv".format(qid)
	plot_df = pd.read_csv(plot_file, index_col=0)

	# # missing index
	missing_index = ["mix-_-Sm-Fe9-Al2-Zn1-_-Zn_10___Al_0-6"]
	plot_df.set_index('index', inplace=True)
	plot_df.drop(missing_index, inplace=True)

	x = plot_df["x_embedd"]
	y = plot_df["y_embedd"]
	index = list(plot_df.index)

	marker_array = plot_df["marker"]
	list_cdict = np.array([get_color_112(k) for k in index])
	alphas = np.array([0.3] * len(index))

	# # find missing index	
	# missing_index = []
	# for idx in index:
	# 	try:
	# 		print (idx, ofm_df.loc[idx, "p1-p1"])
	# 	except Exception as e:
	# 		missing_index.append(idx)
	pca = PCA(n_components=2)
	xy = np.array([x, y]).T
	xy = pca.fit_transform(xy)
	terms = [ 
		"of", "s1-", "s2-",
		"p1-", "d2-", "d5-", "d6-", "d7-", "d10-", "f6-",
		"d10-"
		]
	for i, v in enumerate(pv):
		for term in terms:
			if term in v:
				save_file = savedir+"/ofm_diff/{0}/{1}.pdf".format(term, v)
				z_values = ofm_df.loc[list(index), v]
				# print (z_values)
				if len(set(z_values)) >1:
					# dump_interpolate(x=xy[:, 0], y=xy[:, 1], 
					# 	z_values=z_values,	save_file=save_file,
							# )
					print (len(index))
					print (z_values.shape)
					print (x.shape, y.shape)

					assert x.shape == y.shape
					assert x.shape == z_values.shape

					scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
							z_values=z_values,
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
							vmin=np.nanmin(z_values), vmax=np.nanmax(z_values), vcenter=0
							)


def ofm_diff_correlation():
	ofm_diff_file = "{}/data/SmFe12/ofm_diff/summary.csv".format(ALdir)
	ofm_df = pd.read_csv(ofm_diff_file, index_col=0)
	pv = ofm_df.columns 

	main_savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	savedir = main_savedir + "/query_60"
	plot_file = savedir + "/query_60ipl_plot.csv"
	plot_df = pd.read_csv(plot_file, index_col=0)
	# # missing index
	missing_index = ["mix-_-Sm-Fe9-Al2-Zn1-_-Zn_10___Al_0-6"]
	plot_df.set_index('index', inplace=True)
	plot_df.drop(missing_index, inplace=True)

	index = list(plot_df.index)

	side_file = "{}/data/SmFe12/210829/summary/coarse_relax.csv".format(ALdir)
	side_data = pd.read_csv(side_file, index_col=0)
	print (len(side_data))
	v_SmFe12 = 175
	terms = [ 
		"of", "s1-", "s2-",
		"p1-", "d2-", "d5-", 
		"d6-", "d7-", "d10-", "f6-",
		"d10-"
		]

	# marker_array = ["+" for k in index] # plot_df["marker"]
	family = [get_family(k) for k in index]
	marker_array = np.array([get_marker_112(k) for k in family])
	list_cdict = np.array([get_color_112(k) for k in index]) # 
	alphas = np.array([0.8] * len(index))

	# # plot all pvs
	# for i, v in enumerate(pv):
	# 	for term in terms:
	# 		if term in v:
	# 			save_file = main_savedir+"/ofm_diff_ene/{0}/{1}.pdf".format(term, v)
	# 			y = plot_df.loc[index, "y_obs"].values
	# 			x = ofm_df.loc[index, v].values

	# 			flt = np.where(x==0)[0]
	# 			x_flt = np.delete(x, flt)
	# 			y_flt = np.delete(y, flt)
	# 			cdict_flt = np.delete(list_cdict, flt)
	# 			mk_flt = np.delete(marker_array, flt)

	# 			# scatter_plot_6(x, y, z_values=None, 
	# 			# 	list_cdict=list_cdict, xvlines=None, yhlines=None, 
	# 			#     sigma=None, mode='scatter', lbl=None, name=None, 
	# 			#     s=1, alphas=alphas, title=None,
	# 			#     x_label='x', y_label='y', 
	# 			#     save_file=save_file, interpolate=False, color='blue', 
	# 			#     preset_ax=None, linestyle='-.', marker=marker_array)
	# 			scaler_y = MinMaxScaler().fit(y.reshape(-1,1))

	# 			if len(x_flt) != 0:
	# 				scatter_plot_6(x=x_flt, y=y_flt, 
	# 						z_values=None,
	# 						list_cdict=cdict_flt, 
	# 						xvlines=[0.0], yhlines=[0.0], 
	# 						sigma=None, mode='scatter', lbl=None, name=None, 
	# 						s=60, alphas=alphas, 
	# 						title=save_file.replace(ALdir, ""),
	# 						x_label=v,
	# 						y_label="y_obs", 
	# 						save_file=save_file, scaler_y=scaler_y,
	# 						interpolate=False, cmap="PiYG",
	# 						preset_ax=None, linestyle='-.', marker=mk_flt,
	# 						# vmin=np.nanmin(z_values), vmax=np.nanmax(z_values), vcenter=0
	# 						)

	# #  # violin plot 
	# selected_pv = ["f6-d7", "d10-d10", "p1-d6", "s1-f6"]
	# save_file = main_savedir+"/violin/{0}.pdf".format("|".join(selected_pv))

	# fig = plt.figure(figsize=(12, 8), linewidth=1.0)
	# grid = plt.GridSpec(5, 5, hspace=0.2, wspace=0.2)
	# main_ax = fig.add_subplot(grid[:, :])
	# for i, v in enumerate(selected_pv):
	# 	ene = plot_df.loc[index, "y_obs"].values
	# 	x = ofm_df.loc[index, v].values

	# 	flt = np.where(x==0)[0]
	# 	x_flt = np.delete(x, flt)
	# 	ene_flt = np.delete(ene, flt)
	# 	# # ái for violin always begin with 1
	# 	main_ax.violinplot(dataset=x_flt, positions=[i+1], showmeans=True)
	# main_ax.axhline(y=0.0, xmin=0.0, xmax=len(selected_pv), linestyle='-.', color='grey', alpha=0.7)

	# set_axis_style(ax=main_ax, labels=selected_pv)
	# plt.tight_layout(pad=1.1)
	# makedirs(save_file)
	# print ("Save at:", save_file)
	# plt.savefig(save_file, transparent=False)

	pres = [ 
	   "d10-", 
	   "d7-", 
	   "d5-", 
	   "d2-", 
	   "p1-", 
	   "s2-", 
	   "s1-", 
	   "f6-", 
	   "d6-"
			]
	posts = [ 
		"d6", "f6",
		"s1", "s2", 
		"p1", "d2", "d5", "d7", "d10"
		]
	# save_file = main_savedir+"/violin/all_custom.pdf"
	save_file = main_savedir+"/violin/new/{0}.pdf".format("|".join(pres))

	# for b in all_pres:
		# pres = [b]
		# save_file = main_savedir+"/violin/{0}.pdf".format(b)

	fig = plt.figure(figsize=(8, 10), linewidth=1.0)
	grid = plt.GridSpec(len(pres), len(posts), hspace=0.0, wspace=0.0)

	for row, pre in enumerate(pres):
		ax = fig.add_subplot(grid[row, :], xticklabels=[])
		# ax2 = ax.twinx()

		for col, post in enumerate(posts):
			v = pre + post
			if v in ofm_df.columns:
				ene = plot_df.loc[index, "y_obs"].values
				x = ofm_df.loc[index, v].values
				# volume = side_data.loc[index, "volume"].values - v_SmFe12

				flt = np.where(x==0)[0]
				x_flt = np.delete(x, flt)
				ene_flt = np.delete(ene, flt)
				# volume_flt = np.delete(volume, flt)

				if len(x_flt)!=0:
					violin_parts = ax.violinplot(dataset=x_flt, positions=[col+1.0-0.2], # 
						showmeans=True, vert=True, #showmedians=True
						showextrema=False,  # False
						points=len(x_flt)
						)
					set_color_violin(x_flt=x_flt, violin_parts=violin_parts, pc="red", nc="darkblue")

					violin_parts_2 = ax.violinplot(dataset=ene_flt, positions=[col+1.0+0.2], # 
						showmeans=True, vert=True, #showmedians=True
						showextrema=False,  # False
						points=len(ene_flt)
						)
					set_color_violin(x_flt=ene_flt, violin_parts=violin_parts_2, pc="gray", nc="orange")


					# violin_parts_3 = ax2.violinplot(dataset=volume_flt, positions=[col+1.0+0.2], # 
					# 	showmeans=True, vert=True, #showmedians=True
					# 	showextrema=False,  # False
					# 	points=len(volume_flt)
					# 	)
					# set_color_violin(x_flt=volume_flt, violin_parts=violin_parts_3, 
					# 	pc="red", nc="darkblue")

					# ax.set_ylim(-0.5,0.25)
					# ax.set_ylim(-0.15,0.28)
					# ax2.set_ylim(-0.15,0.28)

					# customize(data=x_flt, ax=ax, positions=[col+1])

		set_axis_style(ax=ax, labels=posts)
		ax.axhline(y=0.0, xmin=0.0, xmax=len(posts), 
			linestyle='-.', color='black', alpha=0.7,
			linewidth=1.0)
		ax.set_ylabel(pre)

		# plt.yscale('symlog')

	ax.set_xlabel("")

	plt.tight_layout(pad=1.1)
	makedirs(save_file)
	print ("Save at:", save_file)
	plt.savefig(save_file, transparent=False)
	# release_mem(fig=fig)



def ofm_diff_correlation_byname():
	ofm_diff_file = "{}/data/SmFe12/ofm_diff/summary.csv".format(ALdir)
	ofm_df = pd.read_csv(ofm_diff_file, index_col=0)
	pv = ofm_df.columns 

	main_savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	savedir = main_savedir + "/query_30"
	plot_file = savedir + "/query_30ipl_plot.csv"
	plot_df = pd.read_csv(plot_file, index_col=0)
	# # missing index
	missing_index = ["mix-_-Sm-Fe9-Al2-Zn1-_-Zn_10___Al_0-6"]
	plot_df.set_index('index', inplace=True)
	plot_df.drop(missing_index, inplace=True)

	index = list(plot_df.index)

	pres = [ 
	   
			]
	posts = [ 
		
		]
	save_file = main_savedir+"/violin/all_custom.pdf"
	# save_file = main_savedir+"/violin/{0}.pdf".format("|".join(pres))

	# for b in all_pres:
		# pres = [b]
		# save_file = main_savedir+"/violin/{0}.pdf".format(b)

	fig = plt.figure(figsize=(6, 8), linewidth=1.0)
	grid = plt.GridSpec(len(pres), len(posts), hspace=0.0, wspace=0.0)

	for row, pre in enumerate(pres):
		ax = fig.add_subplot(grid[row, :], xticklabels=[])
		# ax2 = ax.twinx()

		for col, post in enumerate(posts):
			v = pre + post
			if v in ofm_df.columns:
				ene = plot_df.loc[index, "y_obs"].values
				x = ofm_df.loc[index, v].values
				
				flt = np.where(x==0)[0]
				x_flt = np.delete(x, flt)
				ene_flt = np.delete(ene, flt)

				if len(x_flt)!=0:
					violin_parts = ax.violinplot(dataset=x_flt, positions=[col+1.0-0.2], # 
						showmeans=True, vert=True, #showmedians=True
						showextrema=False,  # False
						points=len(x_flt)
						)

					# norm = colors.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
					set_color_violin(x_flt=x_flt, violin_parts=violin_parts, pc="red", nc="darkblue")

					violin_parts_2 = ax.violinplot(dataset=ene_flt, positions=[col+1.0+0.2], # 
						showmeans=True, vert=True, #showmedians=True
						showextrema=False,  # False
						points=len(ene_flt)
						)
					set_color_violin(x_flt=ene_flt, violin_parts=violin_parts_2, pc="gray", nc="orange")

					# ax.set_ylim(-0.5,0.25)
					# ax.set_ylim(-0.15,0.28)
					# ax2.set_ylim(-0.15,0.28)

					# customize(data=x_flt, ax=ax, positions=[col+1])

		set_axis_style(ax=ax, labels=posts)
		ax.axhline(y=0.0, xmin=0.0, xmax=len(posts), 
			linestyle='-.', color='black', alpha=0.7,
			linewidth=1.0)
		ax.set_ylabel(pre)

		# plt.yscale('symlog')

	ax.set_ylabel("")

	plt.tight_layout(pad=1.1)
	makedirs(save_file)
	print ("Save at:", save_file)
	plt.savefig(save_file, transparent=False)
	# release_mem(fig=fig)





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

def A_matrix(A_file, saveat):
	df = pd.read_csv(A_file, index_col=0)
	df = df.sort_values(by=['0'])
	array = df['0'].values
	index = df.index

	scatter_plot(index, array, xvline=None, yhline=None, 
		sigma=None, mode='line', lbl=None, name=None, 
		x_label='x', y_label='y', 
		save_file=saveat, interpolate=False, color='blue', 
		linestyle='-.', marker='o', title=None)

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


	qids = range(1,31) # 5, 10, 15, 20, 25, 30
	all_data = load_data()
	pv = all_data[-1]

	df = pd.DataFrame(index=pv, columns=qids)

	savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	save_score = savedir+"/x_on_embedd_score_qid.csv"
	ref = "y_all_obs"

	# for qid in qids:
	# 	get_embedding_map(qid=qid, all_data=all_data)
	# 	# ofm_diff_map(qid=qid, all_data=all_data)

	# 	qid_dir = savedir + "/query_{}/x_on_embedd_".format(qid)
	# 	ref_file = qid_dir + "/{}.txt".format(ref)
	# 	ref_map = np.loadtxt(ref_file)
	# 	for v in pv:
	# 		sim_score = two_map_sim(
	# 			savedir=qid_dir, ref=ref, ft=v,
	# 			ref_map=ref_map)
	# 		df.loc[v, qid] = sim_score
	# 		df.to_csv(save_score)

	ofm_diff_correlation()


	# qid = 1
	# A_file =  savedir + "/query_{0}/Amatrix_{0}.csv".format(qid)
	# saveat =  savedir + "/query_{0}/Amatrix_{0}.pdf".format(qid)

	# A_matrix(A_file=A_file, saveat=saveat)









		