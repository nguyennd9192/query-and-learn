import sys
import os
from absl import app
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2 as cv

from params import *
from run_experiment import get_savedir, get_savefile
from utils.utils import load_pickle
from utils.plot import scatter_plot, makedirs, get_color_112, get_marker_112, process_name
from matplotlib.backends.backend_agg import FigureCanvas

result_dir = "Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/ofm_subs_Ga123_margin/"


def read_exp_params(k):
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

		# # result_dict
		# #['accuracy', 'selected_inds', 'data_sizes', 'indices', 
		# #'noisy_targets', 'sampler_output']
		# print(result_dict.keys())

		if exp_params["seed"] == 3:
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

def video_for_tunning(ith_trial="000"): 
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

	# # for video
	width = 800
	height = 800
	saveat = result_dir + "/" + filename + "_" + ith_trial +  "/" + "selection_path.mp4"
	out = cv.VideoWriter(saveat, cv.VideoWriter_fourcc(*'MP4V'),30.0,(width,height))


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
			x = df.loc[index_train[selected_inds], x_lbl]
			y = df.loc[index_train[selected_inds], y_lbl]
			 # = [  if (i % 10) == 0 else ""]
			totalFrames = len(y)
			


			for frameNo in range(totalFrames):				
				fig = plt.figure(figsize=(8, 8),constrained_layout=True,dpi=100)
				canvas = FigureCanvas(fig)
				color = get_color_112(index_train[selected_inds[frameNo]])
				name = process_name(input_name=index_train[selected_inds[frameNo]], main_dir=struct_dir)
				marker = get_marker_112(name)

				this_selected = selected_inds[:frameNo]
				x_frame = x[:frameNo]
				y_frame = y[:frameNo]
				
				colors = [get_color_112(k) for k in index_train[this_selected]]
				name = process_name(input_name=index_train[this_selected], main_dir=struct_dir)
				markers = [get_marker_112(k) for k in name]

				scatter_plot(x=x_frame, y=y_frame, xvline=None, yhline=None, 
						sigma=None, name=None, # name
						mode='scatter', lbl="m:{0}__c{1}".format(m, c),
						x_label=x_lbl, y_label=y_lbl, 
						preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
						color=colors, marker=markers
						)

				# test_idx = list(set(df.index) - set(index_train[selected_inds]))
				print(frameNo)
				x_test = df.loc[test_idx, x_lbl]
				y_test = df.loc[test_idx, y_lbl]
				colors = ["k" for k in test_idx]
				# name = test_idx
				name = process_name(input_name=test_idx, main_dir=struct_dir)
				markers = ["*" for k in name]

				title = result_dir.replace("/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/", "") + \
																	 "/\n" + filename + "_" + ith_trial 
				scatter_plot(x=x_test, y=y_test, xvline=None, yhline=None, 
						sigma=None, name=None, title=title,
						mode='scatter', lbl="m:{0}__c{1}".format(m, c),
						x_label=x_lbl, y_label=y_lbl, 
						preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
						color=colors, marker=markers
						)
				canvas.draw()
				plt.close()

				X = np.array(canvas.renderer.buffer_rgba())
				new_X = np.delete(X.reshape(-1,4),3,1)
				new_X = new_X.reshape(X.shape[0],X.shape[1],-1)

				# print(new_X)
				out.write(new_X)

				plt.close()
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

			title = result_dir.replace("/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/", "") + \
																 "/\n" + filename + "_" + ith_trial 
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
	video_for_tunning(ith_trial="000") # # to prepare selection video



 