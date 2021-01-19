import numpy as np
import matplotlib.pyplot as plt
import time, gc, os
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.collections import LineCollection

import matplotlib as mpl
from general_lib import get_basename

from matplotlib.colors import Normalize



axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
size_text = 10
alpha_point = 0.8
size_point = 100


def release_mem(fig):
	fig.clf()
	plt.close()
	gc.collect()

def ax_setting():
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=15)
	plt.tick_params(axis='y', which='major', labelsize=15)

def ax_setting_3d(ax):
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	ax.xaxis._axinfo["grid"]['color'] =  "w"
	ax.yaxis._axinfo["grid"]['color'] =  "w"
	ax.zaxis._axinfo["grid"]['color'] =  "w"

	# ax.set_xticks([])
	# ax.set_zticks([])

	ax.tick_params(axis='x', which='major', labelsize=15)
	ax.tick_params(axis='y', which='major', labelsize=15)
	ax.tick_params(axis='z', which='major', labelsize=15)

	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False


	ax.xaxis.pane.set_edgecolor('w')
	ax.yaxis.pane.set_edgecolor('w')
	ax.zaxis.pane.set_edgecolor('w')

	ax.zaxis.set_rotate_label(False)

	# ax.view_init(45, 120) # good for Tc
	plt.tight_layout(pad=1.1)

def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))

def set_plot_configuration(x, y, rmin=None, rmax=None):
	if rmin is None and rmax is None:
		y_min = min([min(x), min(y)])
		y_max = max([max(x), max(y)])
		y_mean = (y_max + y_min) / 2.0
		y_std = (y_max - y_mean) / 2.0
		y_min_plot = y_mean - 2.4 * y_std
		y_max_plot = y_mean + 2.4 * y_std
	else:
		y_min_plot = rmin
		y_max_plot = rmax

	# threshold = 0.1
	# plt.plot(x_ref, x_ref * (1 + threshold), 'g--', label=r'$\pm 10 \%$')
	# plt.plot(x_ref, x_ref * (1 - threshold), 'g--', label='')

	plt.ylim([y_min_plot, y_max_plot])
	plt.xlim([y_min_plot, y_max_plot])

	plt.tick_params(axis='x', which='major', labelsize=16)
	plt.tick_params(axis='y', which='major', labelsize=16)

	plt.legend(loc=2, fontsize=18)

	return y_min_plot, y_max_plot

def plot_regression(x, y, tv, dim, rmin=None, rmax=None, name=None, 
	label=None, title=None, point_color='blue', save_file=None):
	
	fig = plt.figure(figsize=(8, 8))
	plt.scatter(x, y, s=size_point, alpha=alpha_point, c=point_color, label=label,
		edgecolor="black")
	y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y, rmin=rmin, rmax=rmax)
	x_ref = np.linspace(y_min_plot, y_max_plot, 100)
	plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)

	if name is not None:
	   for i in range(len(name)):
		   plt.annotate(str(name[i]), xy=(x[i], y[i]), size=size_text)

	plt.ylabel(r'%s predicted (%s)' % (tv, dim), **axis_font)
	plt.xlabel(r'%s observed (%s)' % (tv, dim), **axis_font)
	plt.title(title, **title_font)
	plt.tight_layout(pad=1.1)
	
	if save_file is not None:
		makedirs(save_file)

		plt.savefig(save_file, transparent=False)
		print ("Save at: ", save_file)
		# release_mem(fig=fig)

def joint_plot(x, y, xlabel, ylabel, save_at, is_show=False):
	fig = plt.figure(figsize=(20, 20))
	# sns.set_style('ticks')
	sns.plotting_context(font_scale=1.5)
	this_df = pd.DataFrame()
	
	this_df[xlabel] = x
	this_df[ylabel] = y

	ax = sns.jointplot(this_df[xlabel], this_df[ylabel],
					kind="kde", shade=True, color='orange').set_axis_labels(xlabel, ylabel)

	ax = ax.plot_joint(plt.scatter,
				  color="g", s=40, edgecolor="white")
	# ax.scatter(x, y, s=30, alpha=0.5, c='red')
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)
	# plt.xlabel(r'%s' %xlabel, **axis_font)
	# plt.ylabel(r'%s' %ylabel, **axis_font)
	# plt.title(title, **self.axis_font)

	# plt.set_tlabel('sigma', **axis_font)
	# ax_setting()
	plt.tight_layout()
	if not os.path.isdir(os.path.dirname(save_at)):
		os.makedirs(os.path.dirname(save_at))
	plt.savefig(save_at)
	if is_show:
		plt.show()

	print ("Save file at:", "{0}".format(save_at))
	release_mem(fig)

def ax_scatter(ax, x, y, marker, list_cdict, x_label=None, y_label=None, 
			name=None, alphas=None, save_at=None, plt_mode="2D"):
	n_points = len(x)

	if alphas is None:
		alphas = [0.8] * n_points

	if plt_mode == "2D":
		for i in range(n_points):
			_cdict = list_cdict[i]
			if len(_cdict.keys()) == 1:
				ax.scatter(x[i], y[i], s=120, 
					alpha=alphas[i], marker=marker[i], 
					c=list(_cdict.keys())[0], edgecolor="black")
			else:
				plt_half_filled(ax=ax, x=x[i], y=y[i], 
					cdict=_cdict, alpha=alphas[i])

		if name is not None:
			for i in range(n_points):
				ax.annotate(name[i], xy=(x[i], y[i]), size=12 )
	if "3D" in plt_mode:
		for i in range(n_points):
			_cdict = list_cdict[i]
			if len(_cdict.keys()) == 1:
				ax.scatter(x[i], y[i], 0, s=120, 
					alpha=alphas[i], marker=marker[i], 
					c=list(_cdict.keys())[0], edgecolor="black")
			else:
				plt_half_filled(ax=ax, x=x[i], y=y[i], 
					cdict=_cdict, alpha=alphas[i])

		if name is not None:
			for i in range(n_points):
				ax.annotate(name[i], xy=(x[i], y[i]), size=12 )


	ax_setting()

	ax.set_xlabel(x_label, **axis_font)
	ax.set_ylabel(y_label, **axis_font)
	if save_at is not None:
		makedirs(save_at)
		plt.savefig(save_at)
		plt.close()

def scatter_plot(x, y, xvline=None, yhline=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o', title=None):
	if preset_ax is not None:
		fig = plt.figure(figsize=(8, 8))

	if 'scatter' in mode:
		n_points = len(x)
		for i in range(n_points):
			plt.scatter(x[i], y[i], s=80, alpha=0.8, 
				marker=marker[i], 
				c=color[i], edgecolor="black") # brown
			# plt.scatter(x[i], y[i], s=80, alpha=0.8, 
			# 	marker=marker[i], 
			# 	c=color[i], edgecolor="white") # brown

	if 'line' in mode:
		plt.plot(x, y,  marker=marker, linestyle=linestyle, color=color,
		 alpha=1.0, label=lbl, markersize=10, mfc='none')

	if xvline is not None:
		plt.axvline(x=xvline, linestyle='-.', color='black')
	if yhline is not None:
		plt.axhline(y=yhline, linestyle='-.', color='black')

	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			plt.annotate(name[i], xy=(x[i], y[i]), size=size_text, c="yellow")
		
	plt.title(title, **title_font)
	plt.ylabel(y_label, **axis_font)
	plt.xlabel(x_label, **axis_font)
	ax_setting()

	if preset_ax is not None:
		# min_y, max_y = np.min(y), np.max(y)
		# dy = max_y - min_y
		# min_y -= 0.2*dy
		# max_y += 0.2*max_y
		# plt.ylim([min_y, max_y])

		# plt.grid(linestyle='--', color="gray", alpha=0.8)
		plt.legend(prop={'size': 16})
		makedirs(save_file)
		plt.savefig(save_file)
		release_mem(fig=fig)


def scatter_plot_2(x, y, color_array=None, xvline=None, yhline=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):


	fig = plt.figure(figsize=(8, 8), linewidth=1.0)
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	


	main_ax = sns.kdeplot(x, y,
			 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
			 cmap='Oranges',
			 shade=True, shade_lowest=True,
			 fontsize=10, ax=main_ax, linewidths=1,
			 vertical=True)
	# main_ax.legend(lbl, 
	# 	loc='lower left', fontsize=18,
	# 	bbox_to_anchor=(1.05, 1.05, ),  borderaxespad=0)
	plt.title(lbl, **title_font)

	if color_array is None:
		main_ax.scatter(x, y, s=80, alpha=0.8, marker=marker, c=color, edgecolor="white")
	else:
		cnorm = Normalize(vmin=min(color_array), vmax=max(color_array) )
		main_plot = main_ax.scatter(x, y, s=80, alpha=0.8, marker=marker, 
			c=color_array, cmap='BuPu',
			edgecolor="white")
		fig.colorbar(main_plot, ax=main_ax)
		# main_ax.colorbar()

	# main_ax.axvline(x=xvline, linestyle='-.', color='black')
	# main_ax.axhline(y=yhline, linestyle='-.', color='black')

	main_ax.set_xlabel(x_label, **axis_font)
	main_ax.set_ylabel(y_label, **axis_font)
	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

	# x_hist.hist(x, c='orange', linewidth=1)
	# y_hist.hist(y, c='orange', linewidth=1)

	# # x-axis histogram
	sns.distplot(x, 
		bins=100, 
		ax=x_hist, hist=False,
		kde_kws={"color": "black", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "grey"},
		vertical=False, norm_hist=False)
	l1 = x_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	x_hist.fill_between(x1, y1, color="orange", alpha=0.3)

	# # y-axis histogram
	sns.distplot(y, 
		bins=100, 
		ax=y_hist, hist=False,
		kde_kws={"color": "black", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "grey"},
		vertical=True, norm_hist=False)
	l1 = y_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	y_hist.fill_between(x1, y1, color="orange", alpha=0.3)

	plt.setp(x_hist.get_xticklabels(), visible=False)
	plt.setp(y_hist.get_yticklabels(), visible=False)
	plt.tight_layout(pad=1.1)

	makedirs(save_file)
	plt.savefig(save_file, transparent=False, bbox_inches="tight")
	print ("Save at: ", save_file)
	release_mem(fig=fig)

def get_ratio(index, element):
	if "/feature/" in index:
		index = index[index.find("/feature/")+len("/feature/"):]
	if "/ofm1_no_d/" in index:
		index = index[:index.find("/ofm1_no_d/")]
	pos = index.find(element)
	# print (index, element)

	r = int(index[pos+2:pos+3])
	# print (index, element, r)
	return r

def get_color_112(index):
	# c = "black"
	colors = dict()
	ratios = []

	state_subs = 0
	cdicts = dict({"Ga":"purple", "Mo":"red", "Zn":"orange", 
		"Co":"brown", "Cu":"blue", "Ti":"cyan", "Al":"yellow"})
	index = index.replace("CuAlZnTi_", "")
	if "mix" in index:
		for element, color in cdicts.items():
			if element in index:
				colors[color] = get_ratio(index=index, element=element)
	else:
		for element, color in cdicts.items():
			if element in index:
				colors[color] = "full"
	if len(colors.keys()) == 4:
		# c = "yellow"
		print ("Here: ", index)
		print ("Colors: ", colors)
	#normalize item number values to colormap
	# norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)

	#colormap possible values = viridis, jet, spectral
	# rgba_color = cm.jet(norm(400),bytes=True) 
	return colors

def get_marker_112(index):
	m = "|"
	if "1-11-1" in index:
		m = "s"
	if "1-10-2" in index:
		m = "o"
	if "1-9-3" in index:
		m = "v"
	if "2-23-1" in index:
		m = "X"
	if "2-22-2" in index:
		m = "p"
	if "2-21-3" in index:
		m = "^"
	return m

def get_family(index):
	if "Sm-Fe9" in index:
		f = "1-9-3"
	elif "Sm-Fe10" in index:
		f = "1-10-2"
	elif "Sm-Fe11" in index:
		f = "1-11-1"
	elif "Sm2-Fe23" in index:
		f = "2-23-1"
	elif "Sm2-Fe22" in index:
		f = "2-22-2"
	elif "Sm2-Fe21" in index:
		f = "2-21-3"
	elif "2-22-2" in index:
		f = "2-22-2"
	else:
		print(index)
	return f


def process_name(input_name, main_dir):
	name = [k.replace(main_dir, "")	for k in input_name]
	name = [k.replace("Sm-Fe11-M", "1-11-1") for k in name]
	name = [k.replace("Sm-Fe10-M2", "1-10-2") for k in name]
	name = [k.replace("Sm-Fe10-Ga2", "1-10-2") for k in name]
	name = [k.replace("Sm2-Fe23-M", "2-23-1") for k in name]
	name = [k.replace("Sm2-Fe23-Ga", "2-23-1") for k in name]
	name = [k.replace("Sm2-Fe22-M2", "2-22-2") for k in name]
	name = [k.replace("Sm2-Fe22-Ga2", "2-22-2") for k in name]
	name = [k.replace("Sm2-Fe21-M3", "2-21-3") for k in name]
	name = [k.replace("Sm2-Fe21-Ga3_std6000", "2-21-3") for k in name]
	name = [k.replace("_wyckoff_2", "") for k in name]
	name = [k.replace("_wyckoff", "").replace("*", "").replace("_", "") for k in name]

	wck_idx = [k[k.find("/"):] for k in name]
	system_name = [k[:k.find("/")] for k in name]
	wck_idx = [''.join(i for i in k if not i.isdigit()) for k in wck_idx]
	name = [i + j for i, j in zip(system_name, wck_idx)]
	name = [k.replace("CuAlZnTi", "") for k in name]

	# name = [k.replace("Mo", "").replace("Ti", "").replace("Al", "") for k in name]
	# name = [k.replace("Cu", "").replace("Ga", "").replace("Zn", "") for k in name]
	return name


def scatter_plot_3(x, y, color_array=None, xvlines=None, yhlines=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	s=100, alpha=0.8, title=None,
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):

	fig = plt.figure(figsize=(8, 8), linewidth=1.0)
	sns.kdeplot(x, y,
			 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
			 cmap='Oranges',
			 shade=True, shade_lowest=False,
			 fontsize=10, linewidths=1,
			 vertical=True)
	# if color_array is None:
	#     plt.scatter(x, y, s=s, alpha=alpha, marker=marker, c=color, 
	#         edgecolor="black")
	# elif isinstance(marker, list):

	if type(s) == float:
		sizes = [s] * len(x)
	else:
		sizes = s
	for _m, _c, _x, _y, _s in zip(marker, color_array, x, y, sizes):
		if _c == "orange":
			plt.scatter(_x, _y, s=_s, marker=_m, c=_c, alpha=0.2, edgecolor="black")# "black"
		else: 
			plt.scatter(_x, _y, s=_s, marker=_m, c=_c, alpha=0.7, edgecolor="black")

	for _m, _c, _x, _y in zip(marker, color_array, x, y):
		if _c == "blue": 
			plt.scatter(_x, _y, s=_s, marker=_m, c=_c, alpha=0.1, edgecolor="black")

	# else:
	#     main_plot = plt.scatter(x, y, s=150, alpha=0.8, marker=marker, 
	#         c=color_array, cmap='viridis',
	#         edgecolor="white")
	#     fig.colorbar(main_plot)

	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)
	for spine in plt.gca().spines.values():
		spine.set_visible(False)
	plt.tick_params(top='off', bottom='off', left='off', right='off', 
		labelleft='off', labelbottom='off')
	plt.tight_layout(pad=1.1)
	sns.set_style(style='white') 


	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)
	release_mem(fig=fig)


def scatter_plot_4(x, y, color_array=None, xvlines=None, yhlines=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	s=100, alphas=0.8, title=None,
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):


	fig = plt.figure(figsize=(8, 8), linewidth=1.0)
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	
	sns.set_style(style='white') 

	# main_ax.legend(lbl, 
	#   loc='lower left', fontsize=18,
	#   bbox_to_anchor=(1.05, 1.05, ),  borderaxespad=0)
	# plt.title(title, **title_font)

	# if color_array is None:
	#     plt.scatter(x, y, s=s, alpha=alpha, marker=marker, c=color, 
	#         edgecolor="black")
	# elif isinstance(marker, list):
	main_ax = sns.kdeplot(x, y,
			 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
			 cmap='Oranges',
			 shade=True, shade_lowest=False,
			 fontsize=10, ax=main_ax, linewidths=1,
			 # vertical=True
			 )

	for _m, _c, _x, _y, _a in zip(marker, color_array, x, y, alphas):
		main_ax.scatter(_x, _y, s=s, marker=_m, c=_c, alpha=_a, edgecolor="black")

	

	for xvline in xvlines:
	  main_ax.axvline(x=xvline, linestyle='-.', color='black')
	for yhline in yhlines:
	  main_ax.axhline(y=yhline, linestyle='-.', color='black')

	main_ax.set_xlabel(x_label, **axis_font)
	main_ax.set_ylabel(y_label, **axis_font)
	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

	# x_hist.hist(x, c='orange', linewidth=1)
	# y_hist.hist(y, c='orange', linewidth=1)
	# red_idx = np.where((np.array(color)=="red"))[0]


	# # x-axis histogram
	sns.distplot(x, bins=100, ax=x_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=False, norm_hist=True)
	l1 = x_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	x_hist.fill_between(x1, y1, color="orange", alpha=0.3)

	# sns.distplot(x[red_idx], bins=100, ax=x_hist, hist=False,
	# 	kde_kws={"color": "blue", "lw": 1},
	# 	# shade=True,
	# 	# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
	# 	vertical=False, norm_hist=True)
	# l1 = x_hist.lines[0]
	# x1 = l1.get_xydata()[:,0]
	# y1 = l1.get_xydata()[:,1]
	# x_hist.fill_between(x1, y1, color="blue", alpha=0.3)

	# # y-axis histogram
	sns.distplot(y, bins=100, ax=y_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=True, norm_hist=True)
	l1 = y_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	y_hist.fill_between(x1, y1, color="orange", alpha=0.3)


	# sns.distplot(y[red_idx], bins=100, ax=y_hist, hist=False,
	# 	kde_kws={"color": "blue", "lw": 1},
	# 	# shade=True,
	# 	# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
	# 	vertical=True, norm_hist=True)
	# l1 = y_hist.lines[0]
	# x1 = l1.get_xydata()[:,0]
	# y1 = l1.get_xydata()[:,1]
	# y_hist.fill_between(x1, y1, color="blue", alpha=0.3)



	plt.setp(x_hist.get_xticklabels(), visible=False)
	plt.setp(y_hist.get_yticklabels(), visible=False)
	plt.tight_layout(pad=1.1)

	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)
	release_mem(fig=fig)


def scatter_plot_5(x, y, list_cdict=None, xvlines=None, yhlines=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	s=100, alphas=0.8, title=None,
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):


	fig = plt.figure(figsize=(8, 8), linewidth=1.0)
	grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
	main_ax = fig.add_subplot(grid[1:, :-1])
	y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
	
	sns.set_style(style='white') 

	
	main_ax = sns.kdeplot(x, y,
			 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
			 cmap='Oranges',
			 shade=True, shade_lowest=False,
			 fontsize=10, ax=main_ax, linewidths=1,
			 # vertical=True
			 )

	for _m, _cdict, _x, _y, _a in zip(marker, list_cdict, x, y, alphas):
		if len(_cdict.keys()) == 1:
			print ("this color:", _cdict.keys())
			main_ax.scatter(_x, _y, s=s, 
				marker=_m, c=list(_cdict.keys())[0], 
				alpha=_a, edgecolor="black")
		else:
			plt_half_filled(ax=main_ax, x=_x, y=_y, 
				cdict=_cdict, alpha=_a
				)

	for xvline in xvlines:
	  main_ax.axvline(x=xvline, linestyle='-.', color='black')
	for yhline in yhlines:
	  main_ax.axhline(y=yhline, linestyle='-.', color='black')

	main_ax.set_xlabel(x_label, **axis_font)
	main_ax.set_ylabel(y_label, **axis_font)
	if name is not None:
		for i in range(len(x)):
			main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

	# # x-axis histogram
	sns.distplot(x, bins=100, ax=x_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=False, norm_hist=True)
	l1 = x_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	x_hist.fill_between(x1, y1, color="orange", alpha=0.3)

	# # y-axis histogram
	sns.distplot(y, bins=100, ax=y_hist, hist=False,
		kde_kws={"color": "grey", "lw": 1},
		# shade=True,
		# hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
		vertical=True, norm_hist=True)
	l1 = y_hist.lines[0]
	x1 = l1.get_xydata()[:,0]
	y1 = l1.get_xydata()[:,1]
	y_hist.fill_between(x1, y1, color="orange", alpha=0.3)


	plt.setp(x_hist.get_xticklabels(), visible=False)
	plt.setp(y_hist.get_yticklabels(), visible=False)
	plt.tight_layout(pad=1.1)

	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)
	release_mem(fig=fig)


def show_one_rst(y, y_pred, ax, y_star_ax, ninst_ax, pos_x, color, is_shown_tails=True):
	
	y_star_ax.grid(which='both', linestyle='-.')
	y_star_ax.grid(which='minor', alpha=0.2)
	ax.grid(which='both', linestyle='-.')
	ax.grid(which='minor', alpha=0.2)
	ninst_ax.grid(which='both', linestyle='-.')
	ninst_ax.grid(which='minor', alpha=0.2)

	error = np.abs(y - y_pred)
	mean = np.mean(error)
	y_min = np.min(y)
	flierprops = dict(markerfacecolor='k', marker='.')
	bplot = ax.boxplot(x=error, vert=True, #notch=True, 
		# sym='rs', # whiskerprops={'linewidth':2},
		positions=[pos_x], patch_artist=True,
		widths=0.1, meanline=True, flierprops=flierprops,
		showfliers=False, showbox=True, showmeans=False)
	# ax.text(pos_x, mean, round(mean, 2),
	# 	horizontalalignment='center', size=14, 
	# 	color=color, weight='semibold')
	patch = bplot['boxes'][0]
	patch.set_facecolor(color)

	# # midle axis
	if is_shown_tails:
		y_star_ax.scatter([pos_x], [y_min], s=100, marker="+", 
			c=color, alpha=1.0, edgecolor="black")

		# bplot = y_star_ax.boxplot(x=y, vert=True, #notch=True, 
		# 		# sym='rs', # whiskerprops={'linewidth':2},
		# 		positions=[pos_x], patch_artist=True,
		# 		widths=0.1, meanline=True, #flierprops=flierprops,
		# 		showfliers=True, showbox=True, showmeans=True
		# 		)
		# y_star_ax.text(pos_x, y_min, round(y_min, 2),
		# 	horizontalalignment='center', size=14, 
		# 	color=color, weight='semibold')
		# patch = bplot['boxes'][0]
		# patch.set_facecolor(color)

		ninst_ax.scatter([pos_x], [len(y)], s=100, marker="+", 
			c=color, alpha=1.0, edgecolor="black")

	return ax, y_star_ax, mean, y_min


def plot_hist(x, ax, x_label, y_label, 
	save_at=None, label=None, nbins=50):
	if save_at is not None:
		fig = plt.figure(figsize=(16, 16))

	# hist, bins = np.histogram(x, bins=300, normed=True)
	# xs = (bins[:-1] + bins[1:])/2

	# plt.bar(xs, hist,  alpha=1.0)
	# y_plot = hist
	y_plot, x_plot, patches = ax.hist(x, bins=nbins, histtype='stepfilled', # step, stepfilled, 'bar', 'barstacked'
										density=True, label=label, log=False,  
										color='black', #edgecolor='none',
										alpha=1.0, linewidth=2)

	# X_plot = np.linspace(np.min(x), np.max(x), 1000)[:, np.newaxis]
	# kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(x.reshape(-1, 1))
	# log_dens = kde.score_samples(X_plot)
	# plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
	# plt.text(-3.5, 0.31, "Gaussian Kernel Density")

	#plt.xticks(np.arange(x_min, x_max, (x_max - x_min) / 30), rotation='vertical', size=6)
	# plt.ylim([1, np.max(y_plot)*1.01])
	# plt.legend()
	ax.set_ylabel(y_label, **axis_font)
	ax.set_xlabel(x_label, **axis_font)
	ax_setting()

	if save_at is not None:
		makedirs(save_at)
		plt.savefig(save_at)
		print ("Save file at:", "{0}".format(save_at))
		release_mem(fig)


def ax_surf(xi, yi, zi, label, mode="2D"):
	fig = plt.figure(figsize=(10, 8))

	if mode == "2D":
		ax = fig.add_subplot(1, 1, 1)
		cs = ax.contourf(xi,yi,zi, levels=20, cmap="Greys")
		cbar = fig.colorbar(cs, label="acq_val") 

	if mode == "3D":
		ax = Axes3D(fig)
		surf = ax.plot_surface(xi,yi,zi, alpha=0.5, cmap="jet", # cmap=cm.hot, #color="orange", 
			shade=False, rcount=500, ccount=500, 
			linewidth=0.1, linestyle="-", antialiased=False)
		# surf = ax.plot_trisurf(xi, yi, zi, alpha=0.2, cmap="jet", 
		# 	linewidth=0.1, vmin=min(zi.ravel()), vmax=max(zi.ravel()))

		surf._facecolors2d = surf._facecolors3d
		surf._edgecolors2d = surf._edgecolors3d
		ax.set_axis_off()
		ax.view_init(64, -147) # 60, 60, 37, -150
		ax_setting_3d(ax=ax)
		fig.patch.set_visible(False)

	if mode == "3D_patch":
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		print (xi.shape, yi.shape, zi.shape)
		xi_rv, yi_rv, zi_rv = xi.ravel(), yi.ravel(), zi.ravel()

		vmin, vmax = max(zi_rv)/2, max(zi_rv)# min(zi_rv), max(zi_rv)
		norm = plt.Normalize(vmin, vmax)

		n_points = 10
		for _xi, _yi, _zi in zip(xi, yi, zi):
			# line = art3d.Line3D(*zip((_xi, _yi, 0), (_xi, _yi, _zi)),
			#  marker="o", markersize=1.0, markevery=(1, 1), linewidth=0.5, 
			#  linestyle="-.", color="black", alpha=0.2 )
			# ax.add_line(line)

			x = np.array([_xi]*n_points)
			y = np.array([_yi]*n_points)
			z = np.linspace(0, _zi, n_points)

			points = np.array([x, y, z]).transpose().reshape(-1,1,3)
			print (points.shape)
			segs = np.concatenate([points[:-1],points[1:]],axis=1)

			lc = Line3DCollection(segs, cmap=plt.cm.hsv, norm=norm, alpha=0.2)
			lc.set_array(z)
			ax.add_collection(lc)


		# ax.set_axis_off()
		ax.view_init(-114, 33) # 60, 60
		ax_setting_3d(ax=ax)
		fig.patch.set_visible(False)

	plt.tight_layout(pad=1.1)
	# plt.legend(loc="upper left", prop={'size': 16})
	# ax.set_xlabel(x_lbl, **axis_font)
	# ax.set_ylabel(y_lbl, **axis_font)
	# ax.set_zlabel(z_lbl, **axis_font, rotation=90)

	# plt.show()

	return ax


# def plt_half_filled(ax, x, y, cdict, alpha):
# 	rot = 30
# 	_sorted_cdict = {k: v for k, v in sorted(cdict.items(), key=lambda item: item[1])}
# 	# small_color, small_ratio = _sorted_cdict[0]
# 	# big_color, big_ratio = _sorted_cdict[1]

# 	# try:
# 	small_color, big_color = list(_sorted_cdict.keys())
# 	small_ratio, big_ratio = list(_sorted_cdict.values())
# 	# except Exception as e:
# 	# 	print (_sorted_cdict)
	

# 	small_angle = 360 * small_ratio / (small_ratio + big_ratio)
# 	# print (small_ratio, big_ratio)
# 	# print (small_color, big_color)
# 	# if z is None:
# 	HalfA = mpl.patches.Wedge((x, y), 0.01, alpha=alpha, 
# 		theta1=0-rot,theta2=small_angle-rot, color=small_color, 
# 		edgecolor="black")
# 	HalfB = mpl.patches.Wedge((x, y), 0.01, alpha=alpha,
# 		theta1=small_angle-rot,theta2=360-rot, color=big_color,
# 		edgecolor="black")
# 	# else:
# 	# 	HalfA = mpl.patches.Wedge((x, y, z), 0.01, alpha=alpha, 
# 	# 		theta1=0-rot,theta2=small_angle-rot, color=small_color, 
# 	# 		edgecolor="black")
# 	# 	HalfB = mpl.patches.Wedge((x, y, z), 0.01, alpha=alpha,
# 	# 		theta1=small_angle-rot,theta2=360-rot, color=big_color,
# 	# 		edgecolor="black")
# 	ax.add_artist(HalfA)
# 	ax.add_artist(HalfB)


def plt_half_filled(ax, x, y, cdict, alpha):

	# small_ratio, big_ratio = sorted(cdict.values())
	# small_color, big_color = sorted(cdict, key=cdict.get)
	color1, color2 = sorted(cdict.keys())
	ratio1, ratio2 = cdict[color1], cdict[color2]
	

	angle1 = 360 * ratio1 / (ratio1 + ratio2)
	
	if angle1 == 180:
		# # for 1-1 composition, e.g. Al1-Ti1
		rot = 90
	elif angle1 < 180:
		rot = 30
	else:
		rot = 150


	# print (small_ratio, big_ratio)
	# print (small_color, big_color)
	# if z is None:
	HalfA = mpl.patches.Wedge((x, y), 0.01, alpha=alpha, 
		theta1=0-rot,theta2=angle1-rot, facecolor=color1, 
		lw=1.5,
		edgecolor="black")
	HalfB = mpl.patches.Wedge((x, y), 0.02, alpha=alpha,
		theta1=angle1-rot,theta2=360-rot, facecolor=color2,
		lw=1.5,
		edgecolor="black")
	# else:
	# 	HalfA = mpl.patches.Wedge((x, y, z), 0.01, alpha=alpha, 
	# 		theta1=0-rot,theta2=small_angle-rot, color=small_color, 
	# 		edgecolor="black")
	# 	HalfB = mpl.patches.Wedge((x, y, z), 0.01, alpha=alpha,
	# 		theta1=small_angle-rot,theta2=360-rot, color=big_color,
	# 		edgecolor="black")
	ax.add_artist(HalfA)
	ax.add_artist(HalfB)

def test_half_filled():
	# mport matplotlib.pyplot as plt
	# import matplotlib as mpl

	# plt.figure(1)
	# ax=plt.gca()
	# rot = 30

	# # 1:1 180:180, 1:2 120:240, 2:1 240:120
	# ratio = 120 # 120, 180, 240
	# # for i in x:
	# i = 0
	# HalfA = mpl.patches.Wedge((i, i), 5,
	# 	theta1=0-rot,theta2=ratio-rot, color='r')
	# HalfB = mpl.patches.Wedge((i, i), 5,
	# 	theta1=ratio-rot,theta2=360-rot, color='b')
	# # rot=rot+360/len(x)

	# ax.add_artist(HalfA)
	# ax.add_artist(HalfB)

	# ax.set_xlim((-10, 10))
	# ax.set_ylim((-10, 10))
	# plt.savefig("test_half_filled.pdf")
	index =  "mix/Sm-Fe10-Al1-Ga1"
	# colors = get_color_112(index) 
	r = get_ratio(index=index, element="Ga")

	print (r)

def plot_heatmap(matrix, vmin, vmax, save_file, cmap, lines=None, title=None):
	if vmax is None:
		vmax = np.max(matrix)
	if vmin is None:
		vmin = np.min(matrix)
	fig = plt.figure(figsize=(10, 8))

	ax = fig.add_subplot(1, 1, 1)
	ax = sns.heatmap(matrix, cmap=cmap, 
			xticklabels=True,
			yticklabels=True,
			vmax=vmax, vmin=vmin)

	makedirs(save_file)
	if title is None:
		plt.title(get_basename(save_file))
	else:
		plt.title(title)

	if lines is not None:
		ax.hlines(lines, *ax.get_xlim(), colors="white")
		ax.vlines(lines, *ax.get_ylim(), colors="white")


	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)
	release_mem(fig=fig)


if __name__ == "__main__":
	test_half_filled()











