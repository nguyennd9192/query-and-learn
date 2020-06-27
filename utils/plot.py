import numpy as np
import matplotlib.pyplot as plt
import time, gc, os
import pandas as pd
import seaborn as sns
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
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)


def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))

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


def scatter_plot(x, y, xvline=None, yhline=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):
	if preset_ax is not None:
		fig = plt.figure(figsize=(8, 8))

	if 'scatter' in mode:
		plt.scatter(x, y, s=80, alpha=0.8, 
		marker=marker, c=color, edgecolor="white") # brown

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
			plt.annotate(name[i], xy=(x[i], y[i]), size=size_text)
		

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
		main_plot = main_ax.scatter(x, y, s=80, alpha=0.8, marker=marker, 
			c=color_array, cmap='viridis',
			edgecolor="white")
		fig.colorbar(main_plot, ax=main_ax)
		# main_ax.colorbar()

	main_ax.axvline(x=xvline, linestyle='-.', color='black')
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

