import sys
import os
from absl import app
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2 as cv

from params import *
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg
from utils.utils import load_pickle
# from utils.plot import scatter_plot, makedirs, get_color_112, get_marker_112, process_name, ax_scatter
from utils.plot import *
from matplotlib.backends.backend_agg import FigureCanvas
from sklearn.preprocessing import normalize
from matplotlib import gridspec

# result_dir = "Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/ofm_subs_Ga123_margin/"

def read_exp_params(k):
	"""
	key for each experiment
	# key = (FLAGS.dataset, FLAGS.sampling_method, FLAGS.score_method,
	# 						 FLAGS.select_method, m, FLAGS.warmstart_size, FLAGS.batch_size,
	# 						 c, standardize_data, normalize_data, seed)
	"""
	exp_params = {}
	exp_params["dataset"] = k[0]
	exp_params["sampling_method"] = k[1]
	exp_params["score_method"] = k[2]
	exp_params["select_method"] = k[3]
	exp_params["m"] = k[4]
	exp_params["warmstart_size"] = k[5]
	exp_params["batch_size"] = k[6]
	exp_params["c"] = k[7]
	exp_params["standardize_data"] = k[8]
	exp_params["normalize_data"] = k[9]
	exp_params["seed"] = k[10] # # number of trials for a given c and m

	assert exp_params["dataset"] == FLAGS.dataset
	assert exp_params["score_method"] == FLAGS.score_method
	assert exp_params["select_method"] == FLAGS.select_method
	assert str(exp_params["normalize_data"]) == str(FLAGS.normalize_data)
	# print("seed:", FLAGS.seed, exp_params["trials"])
	assert exp_params["seed"] <= FLAGS.trials

	return exp_params


colors = dict({0.1:"blue", 0.3:"red", 0.5:"yellow", 0.7:"brown", 0.9:"green"})
markers = dict({0.1:"o", 0.3:"p", 0.5:"v", 0.7:"^", 0.9:"s"})
# markers = ["o", "p", "v", "^", "s"]


def main_proc(ith_trial="000"):
	result_dir = get_savedir()
	filename = get_savefile()

	# ith_trial = "000"
	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	print(result_file)

	all_results = load_pickle(result_file)
	# print(all_results.keys())

	fig = plt.figure(figsize=(10, 8))
	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		print("m:", exp_params["m"])
		print("c:", exp_params["c"])

		m, c = exp_params["m"], exp_params["c"]
		accuracies = np.array(result_dict["accuracy"])
		acc_cv_train = np.array(result_dict["cv_train_model"])

		x = np.arange(len(accuracies))
		plot_idx = [k for k in x]
		# print("Here", acc_cv_train)

		scatter_plot(x=x[plot_idx] + 1, y=accuracies[plot_idx], xvline=None, yhline=None, 
				sigma=None, mode='line', lbl="m:{0}__c{1}".format(m, c), name=None, 
				x_label='n update', y_label='accuracy', 
				preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
				 color=colors[m], marker=markers[c]
				)
		# scatter_plot(x=x[plot_idx] + 1, y=acc_cv_train[plot_idx], xvline=None, yhline=None, 
		# 		sigma=None, mode='scatter', lbl="m:{0}__c{1}".format(m, c), name=None, 
		# 		x_label='n update', y_label='accuracy', 
		# 		preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
		# 		 color=colors[m], marker=markers[c]
		# 		)
		# print(result_dict.items())

		models = [k.estimator.get_params() for k in result_dict["save_model"]]
		GSCVs = [k.GridSearchCV.best_score_ for k in result_dict["save_model"]]


		for model, gscv in zip(models, GSCVs):
			print(model)
			print(gscv)

			print("===================")

		print("data_sizes:", result_dict["data_sizes"])
		# print("cv", result_dict["cv_train_model"])

	plt.xticks(x[plot_idx] + 1)
	plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', 
		mode="expand", borderaxespad=0, ncol=3, fontsize=12)
	saveat = result_dir + "/" + filename + "_" + ith_trial +  "/" + "accuracy.pdf"
	print ("saveat:", saveat)
	makedirs(saveat)

	init_data_size = result_key_to_text["data_sizes"][0]
	batch_data_size = result_key_to_text["data_sizes"][1] - init_data_size

	final_data_size = result_key_to_text["data_sizes"][-1]


	text = "\n".join(["{0}: {1}".format(k, v) for k, v in exp_params.items() if k != "m" and k != "c"])
	text += "\nm: Mix weights on ActSamp\n"
	text += "c: percent of rand labels\n"
	# text += "\n".join(["{0}: {1}".format(k, v) for k, v in result_key_to_text.items() if k != "data_sizes"])
	text += "batch data size: " + str(batch_data_size) + "\n"
	# text += "init train size: "  + str(init_data_size)  + " cv: " + str(round(result_key_to_text["cv_train_model"][0], 2)) + "\n"
	# text += "final train size: " + str(final_data_size) + " cv: " + str(round(result_key_to_text["cv_train_model"][-1], 2)) +  "\n"
	text += "test data size: " + str(result_key_to_text["n_test"]) + "\n"
	text += "is test separate: " + str(result_key_to_text["is_test_separate"]) + "\n"
	text += "test_prefix: " + str(result_key_to_text["test_prefix"]) + "\n"

	text += "y_train_info: " + str(result_key_to_text["y_train_info"]) + "\n"
	text += "y_val_info: " + str(result_key_to_text["y_val_info"]) + "\n"
	text += "y_test_info: " + str(result_key_to_text["y_test_info"]) + "\n"

	# org_data_size
	print(result_key_to_text["y_train_info"])
	print(str(result_key_to_text["data_sizes"]))
	# plt.text(35.5, 0.5, text, fontsize=14)

	side_text = plt.figtext(0.91, 0.12, text, bbox=dict(facecolor='white'))
	fig.subplots_adjust(top=0.8)
	# plt.title(text, fontsize=14)
	# plt.tight_layout(pad=1.2)
	# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(saveat, bbox_inches='tight')
		# print("accuracy:", result_dict["accuracy"])
		# break


def video_for_tunning(ith_trial, verbose=True): 
	result_dir = get_savedir()
	filename = get_savefile()
	struct_dir = "/media/nguyen/work/SmFe12_screening/input/origin_struct/"

	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	all_results = load_pickle(result_file)
	print(result_file)

	# # load dtrain file
	# dtrain_file = os.path.join(FLAGS.data_dir, FLAGS.dataset+"/train_"+FLAGS.test_prefix + ".pkl")
	# data = load_pickle(dtrain_file)
	# X_train, y_train = data["data"], data["target"]
	# index_train = data["index"]

	# # # load dtrain file
	# dtest_file = os.path.join(FLAGS.data_dir, FLAGS.dataset+"/test_"+FLAGS.test_prefix + ".pkl")
	# data = load_pickle(dtest_file)
	# X_test, y_test = data["data"], data["target"]
	# test_idx = data["index"]

	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	# max_points, train_size, batch_size, seed_batch = get_othere_cfg(y,max_points,batch_size,warmstart_size)
	# confusion = 0.1
	# all_X = get_train_test(X,y,X_sept_test, y_sept_test,max_points,seed,confusion,seed_batch,data_splits)
	# indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = all_X
	if verbose:
		print("X_train, y_train, index_train, X_test, y_test, test_idx")
		print(X_trval_csv.shape, y_trval_csv.shape, len(index_trval_csv), X_test_csv.shape, y_test_csv.shape, len(test_idx_csv))
		# print("indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise")
		# print(len(indices), X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, y_noise.shape)
	# X_train_csv = normalize(X_train_csv)
	# X_test_csv = normalize(X_test_csv)

	max_y = max(np.abs(np.concatenate((y_test_csv, y_trval_csv))))*1.1

	"""
	selected_inds: work only on X_train data
	"""
	csv_file = os.path.join(FLAGS.data_dir, FLAGS.dataset + ".csv")
	df = pd.read_csv(csv_file, index_col=0)
	df = df.dropna()
	x_lbl = "magmom_pa"
	y_lbl = "energy_substance_pa"

	# x_train_plt = df.loc[index_trval_csv, x_lbl].values
	# y_train_plt = df.loc[index_trval_csv, y_lbl].values
	# x_test_plt = df.loc[test_idx_csv, x_lbl].values
	# y_test_plt = df.loc[test_idx_csv, y_lbl].values


	# # for video
	width = 800
	height = 800
	saveat = result_dir + "/" + filename + "_" + ith_trial +  "/" + "selection_path.mp4"
	makedirs(saveat)
	out = cv.VideoWriter(saveat, cv.VideoWriter_fourcc(*'MP4V'),30.0,(width,height))

	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		print("m:", exp_params["m"])
		print("c:", exp_params["c"])

		# # for a given pair "m", "c", we have an accuracy line
		m, c = exp_params["m"], exp_params["c"]
		accuracies = np.array(result_dict["accuracy"])
		acc_cv_train = np.array(result_dict["cv_train_model"])

		x = np.arange(len(accuracies))
		plot_idx = [k for k in x]
		trace_selected_inds = np.array(result_dict["selected_inds"])

		
		# batch_size = result_dict["data_sizes"][1] - result_dict["data_sizes"][0]
		# print("all_X", result_dict["all_X"])
		# print("batch_size", batch_size)
		# print("exp_params:", exp_params)
		# print("selected_inds:", trace_selected_inds)
		batches = result_dict["batches"]
		min_margins = result_dict["min_margins"]
		print("batches", batches)

		# print("un_shfl_test_idx:", result_dict["un_shfl_test_idx"])
		# print("un_shfl_train_val_idx:", result_dict["un_shfl_train_val_idx"])

		shfl_indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise, idx_train, idx_val, idx_test = result_dict["all_X"]
		# # idx_train, idx_val, idx_test: index of train, test in np matrix after shuffling
		# print("idx_train:", idx_train)
		# print("idx_test:", idx_test)
		n_train = len(y_train)

		# # currently, df hold all train, val and test set. This step to convert back from df to trval + test
		x_trval_plt = df.loc[index_trval_csv, x_lbl].values
		y_trval_plt = df.loc[index_trval_csv, y_lbl].values
		

		if idx_test is not None:
			x_test_plt = df.loc[test_idx_csv[idx_test], x_lbl].values
			y_test_plt = df.loc[test_idx_csv[idx_test], y_lbl].values
			index_test_plt = test_idx_csv[idx_test]
		else:
			# # for separated testset
			x_test_plt = df.loc[test_idx_csv, x_lbl].values
			y_test_plt = df.loc[test_idx_csv, y_lbl].values
			index_test_plt = test_idx_csv
		# # Finish converting back from df to trval + test


		selected_inds = []
		for frameNo, (batch, min_margin) in enumerate(zip(batches, min_margins)):
			fig = plt.figure(figsize=(8, 8)) 
			gs = gridspec.GridSpec(nrows=2,ncols=2,figure=fig,width_ratios=[1, 1]) 
			canvas = FigureCanvas(fig)

			# # must fix error of cannot get "this_batchs"
			# this_batch = trace_selected_inds[frameNo*batch_size:(frameNo+1)*batch_size]
			selected_inds.extend(batch)

			# if frameNo == 15:
			# 	notEnd = False
			# print("selected_inds", selected_inds)
			if len(selected_inds) == 0:
				continue
			if True:
				# ax1 = fig.add_subplot()
				ax1 = plt.subplot(gs[0, 0])
				"""
				Begin plot map properties
				"""
				csv_idx_cvt = shfl_indices[selected_inds]
				x_frame = x_trval_plt[csv_idx_cvt]
				y_frame = y_trval_plt[csv_idx_cvt]
				
				train_colors = [get_color_112(k) for k in index_trval_csv[csv_idx_cvt]]
				train_names = process_name(input_name=index_trval_csv[csv_idx_cvt], main_dir=struct_dir)
				train_markers = [get_marker_112(k) for k in train_names]
				ax_scatter(ax=ax1,x=x_frame,y=y_frame,marker=train_markers,color=train_colors)

				# # for test points
				colors = ["k" for k in index_test_plt]
				name = process_name(input_name=index_test_plt, main_dir=struct_dir)
				markers = ["*" for k in name]
				ax_scatter(ax=ax1,x=x_test_plt,y=y_test_plt,
					marker=markers,color=colors)
				fig.tight_layout(rect=[0, 0.03, 1, 0.95])
			if True:
				"""
				prediction plot
				"""
				ax2 = plt.subplot(gs[1, 0])
				estimator = result_dict["save_model"][frameNo].estimator

				partial_X = X_train[selected_inds]
				partial_y = y_train[selected_inds]

				# # check whether or not csv_idx_cvt point to partial_X
				np.testing.assert_array_equal(partial_X, X_trval_csv[csv_idx_cvt])
				np.testing.assert_array_equal(partial_y, y_trval_csv[csv_idx_cvt])

				# # estimator saved here has already fit with the last selected inds
				# estimator.fit(partial_X, partial_y)

				y_test_pred = estimator.predict(X_test)
				acc = estimator.score(X_test, y_test)
				print("frameNo:", frameNo, "acc:", acc)
				# # train test prediction
				ax2.scatter(y_test, y_test_pred, c="black", marker="*", 
					label="acc: {0}".format(round(acc, 3)))
				ax_scatter(ax=ax2,x=partial_y,y=estimator.predict(partial_X),
					marker=train_markers,color=train_colors)

				lb, ub = -max_y, 0.3
				ref = np.arange(lb, ub, (ub - lb)/100.0)
				ax2.plot(ref, ref, linestyle="-.", c="r")
				ax2.set_xlim(lb, ub) # max_y 
				ax2.set_ylim(lb, ub)
				ax2.set_xlabel("Observed value", **axis_font)
				ax2.set_ylabel("Predicted value", **axis_font)
				title = result_dir.replace(FLAGS.save_dir, "") + "/\n" + filename + "_" + ith_trial
				fig.suptitle(title,  y=0.98)
				fig.tight_layout(rect=[0, 0.03, 1, 0.95])
			if True:
				"""
				min_margin plot
				"""
				ax3 = plt.subplot(gs[0,1])

				acc_pos = range(frameNo)
				ax3.plot(acc_pos,accuracies[:frameNo], "r"
          )
				
			if True:
				ax4 = plt.subplot(gs[1,1])
				pos = np.arange(len(min_margin))

				# # min_margin has the same index with X_train or idx_train
				name_min_margin = np.array(["" for k in range(n_train)])
				color_min_margin = np.array(["yellow" for k in range(n_train)])
				marker_min_margin = np.array(["." for k in range(n_train)])

				name_min_margin[selected_inds] = train_names
				color_min_margin[selected_inds] = train_colors
				marker_min_margin[selected_inds] = train_markers

				ax_scatter(ax=ax4,x=pos,y=min_margin,
					marker=marker_min_margin,color=color_min_margin)
				# ax4.scatter(pos, min_margin, marker=marker_min_margin,color=color_min_margin
    #       )
				canvas.draw()
				rgba_render = np.array(canvas.renderer.buffer_rgba())
				final_frame = np.delete(rgba_render.reshape(-1,4),3,1)
				# print("shape before:", final_frame.shape)
				final_frame = final_frame.reshape(final_frame.shape[0],final_frame.shape[1],-1)
				final_frame = final_frame.reshape(800, 800,-1)
				# print("shape after:", final_frame.shape)
				out.write(final_frame)
				plt.close()
			# if frameNo == 15:
			# 	break
			# break

		out.release()
		cv.destroyAllWindows()
		print("Save at:", saveat)

		break

def selection_path(ith_trial="000"):
	result_dir = get_savedir()
	filename = get_savefile()
	# ith_trial = "000"
	struct_dir = "/media/nguyen/work/SmFe12_screening/input/origin_struct/"

	result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
	print(result_file)

	# # load dtrain file
	dtrain_file = os.path.join(FLAGS.data_dir, FLAGS.dataset+"/train_"+FLAGS.test_prefix + ".pkl")
	data = load_pickle(dtrain_file)
	X_train, y_train = data["data"], data["target"]
	index_train = data["index"]

	# # load dtrain file
	dtest_file = os.path.join(FLAGS.data_dir, FLAGS.dataset+"/test_"+FLAGS.test_prefix + ".pkl")
	data = load_pickle(dtest_file)
	X_test, y_test = data["data"], data["target"]
	test_idx = data["index"]

	all_results = load_pickle(result_file)
	# print(all_results.keys())

	csv_file = os.path.join(FLAGS.data_dir, FLAGS.dataset + ".csv")
	df = pd.read_csv(csv_file, index_col=0)
	df = df.dropna()
	x_lbl = "magmom_pa"
	y_lbl = "energy_substance_pa"
		
	for result_key, result_dict in all_results.items():
		# # "k" of all_results store all setting params 
		if result_key == "tuple_keys":
			continue
		else:
			result_key_to_text = result_dict
		exp_params = read_exp_params(result_key)

		# # result_dict
		# #['accuracy', 'selected_inds', 'data_sizes', 'indices', 
		# #'noisy_targets', 'sampler_output']
		# print(result_dict.keys())

		if exp_params["seed"] == 1:
			print("m:", exp_params["m"])
			print("c:", exp_params["c"])

			m, c = exp_params["m"], exp_params["c"]
			accuracies = np.array(result_dict["accuracy"])
			acc_cv_train = np.array(result_dict["cv_train_model"])

			x = np.arange(len(accuracies))
			plot_idx = [k for k in x]
			selected_inds = np.array(result_dict["selected_inds"])
			print("selected_inds:", selected_inds)
			print("to_name:", index_train[selected_inds])

			# selected_inds = len(df)
			fig = plt.figure(figsize=(8, 8))
			x = df.loc[index_train[selected_inds], x_lbl]
			y = df.loc[index_train[selected_inds], y_lbl]
			colors = [get_color_112(k) for k in index_train[selected_inds]]
			name = process_name(input_name=index_train[selected_inds], main_dir=struct_dir)
			markers = [get_marker_112(k) for k in name]

			name = []
			for i, k in enumerate(selected_inds):
				if (i%10)==0:
					name.append(k)
				else:
					name.append("")


			scatter_plot(x=x, y=y, xvline=None, yhline=None, 
					sigma=None, name=name,
					mode='scatter', lbl="m:{0}__c{1}".format(m, c),
					x_label=x_lbl, y_label=y_lbl, 
					preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
					color=colors, marker=markers
					)

			# test_idx = list(set(df.index) - set(index_train[selected_inds]))
			print(len(test_idx))
			x_test = df.loc[test_idx, x_lbl]
			y_test = df.loc[test_idx, y_lbl]
			colors = ["k" for k in test_idx]
			# name = test_idx
			name = process_name(input_name=test_idx, main_dir=struct_dir)
			markers = ["*" for k in name]

			title = result_dir.replace(FLAGS.save_dir, "") + "/\n" + filename + "_" + ith_trial 
			scatter_plot(x=x_test, y=y_test, xvline=None, yhline=None, 
					sigma=None, name=None, title=title,
					mode='scatter', lbl="m:{0}__c{1}".format(m, c),
					x_label=x_lbl, y_label=y_lbl, 
					preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
					color=colors, marker=markers
					)
		break
	# # plt.xticks(x[plot_idx] + 1)
	# # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', 
	# # 	mode="expand", borderaxespad=0, ncol=3, fontsize=12)
	side_text = plt.figtext(0.91, 0.12, text, bbox=dict(facecolor='white'))
	fig.subplots_adjust(top=0.8)
	plt.title(text, fontsize=14)
	plt.tight_layout(pad=1.2)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(saveat, bbox_inches='tight')
	saveat = result_dir + "/" + filename + "_" + ith_trial +  "/" + "selection_path.pdf"
	print ("saveat:", saveat)
	makedirs(saveat)


if __name__ == "__main__":
	# app.run(main_proc(ith_trial="000"))
	FLAGS(sys.argv)
	# main_proc(ith_trial="000") # # to plot learning curver
	# selection_path(ith_trial="000") # # to plot selection figure 
	video_for_tunning(ith_trial="008") # # to prepare selection video



 