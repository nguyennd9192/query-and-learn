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

def ax_scatter(ax,x,y,marker,color,name=None):
	n_points = len(x)
	for i in range(n_points):
		ax.scatter(x[i], y[i], s=80, alpha=0.8, 
		marker=marker[i], 
 		c=color[i], edgecolor="black") # brown

	if name is not None:
		for i in range(n_points):
			ax.annotate(name[i], xy=(x[i], y[i]), size=12 ) # brown


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

def get_color_112(index):
	c = "yellow"
	if "Ga" in index:
		c = "purple"
	if "Mo" in index:
		c = "red"
	if "Zn" in index:
		c = "orange"
	if "Co" in index:
		c = "brown"
	if "Cu" in index:
		c = "blue"
	if "Ti" in index:
		c = "cyan"
	if "Al" in index:
		c = "green"
	return c

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
    s=100, alpha=0.8, title=None,
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
             vertical=True)
    # main_ax.legend(lbl, 
    #   loc='lower left', fontsize=18,
    #   bbox_to_anchor=(1.05, 1.05, ),  borderaxespad=0)
    # plt.title(title, **title_font)

    # if color_array is None:
    #     plt.scatter(x, y, s=s, alpha=alpha, marker=marker, c=color, 
    #         edgecolor="black")
    # elif isinstance(marker, list):
    for _m, _c, _x, _y in zip(marker, color_array, x, y):
        main_ax.scatter(_x, _y, marker=_m, c=_c, alpha=alpha, edgecolor="black")
    # else:
    #     main_plot = plt.scatter(x, y, s=150, alpha=0.8, marker=marker, 
    #         c=color_array, cmap='viridis',
    #         edgecolor="white")
    #     fig.colorbar(main_plot)

    # for xvline in xvlines:
    #   main_ax.axvline(x=xvline, linestyle='-.', color='black')
    # for yhline in yhlines:
    #   main_ax.axhline(y=yhline, linestyle='-.', color='black')

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
    red_idx = np.where((np.array(color)=="red"))[0]

    n_total = len(x)
    n_blue = len(x)

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

    sns.distplot(x[red_idx], bins=100, ax=x_hist, hist=False,
        kde_kws={"color": "blue", "lw": 1},
        # shade=True,
        # hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
        vertical=False, norm_hist=True)
    # l1 = x_hist.lines[0]
    # x1 = l1.get_xydata()[:,0]
    # y1 = l1.get_xydata()[:,1]
    # x_hist.fill_between(x1, y1, color="blue", alpha=0.3)

    # # y-axis histogram
    sns.distplot(y, bins=100, ax=y_hist, hist=False,
        kde_kws={"color": "orange", "lw": 1},
        # shade=True,
        # hist_kws={"linewidth": 3, "alpha": 0.3, "color": "orange"},
        vertical=True, norm_hist=True)
    l1 = y_hist.lines[0]
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    y_hist.fill_between(x1, y1, color="orange", alpha=0.3)


    sns.distplot(y[red_idx], bins=100, ax=y_hist, hist=False,
        kde_kws={"color": "blue", "lw": 1},
        # shade=True,
        # hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
        vertical=True, norm_hist=True)
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


def plot_hist(x, ax, save_at=None, label=None, nbins=50):

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
	plt.legend()
	plt.ylabel('Probability density', **axis_font)
	plt.xlabel("Value", **axis_font)

	ax_setting()

	if save_at is not None:
		makedirs(save_at)
		plt.savefig(save_at)
		print ("Save file at:", "{0}".format(save_at))
		release_mem(fig)