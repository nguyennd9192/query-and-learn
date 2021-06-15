import numpy as np
import matplotlib.pyplot as plt
import time, gc, os, copy
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.collections import LineCollection

import matplotlib as mpl
from utils.general_lib import get_basename
# from utils import utils

from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.cm as cm

from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 8}
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
    ax.xaxis._axinfo["grid"]['color'] =  "grey"
    ax.yaxis._axinfo["grid"]['color'] =  "grey"
    ax.zaxis._axinfo["grid"]['color'] =  "grey"
    # ax.xaxis._axinfo["grid"]['linestyle'] =  "-"

    # ax.set_xticks([])
    # ax.set_zticks([])

    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='z', which='major', labelsize=15)
    # ax.set_zticks([])

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
    linestyle='-.', marker='o', title=None):
    fig = plt.figure(figsize=(8, 8))

    if 'scatter' in mode:
        n_points = len(x)
        for i in range(n_points):
            plt.scatter(x[i], y[i], s=80, alpha=0.8, 
                marker=marker, 
                c="black", edgecolor="black") # brown, color[i]
            # plt.scatter(x[i], y[i], s=80, alpha=0.8, 
            #   marker=marker[i], 
            #   c=color[i], edgecolor="white") # brown

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


    plt.legend(prop={'size': 16})
    makedirs(save_file)
    plt.savefig(save_file)
    release_mem(fig=fig)


def scatter_plot_2(x, y, z_values=None, color_array=None, xvline=None, yhline=None, 
    sigma=None, mode='scatter', lbl=None, name=None, 
    x_label='x', y_label='y', title=None,
    save_file=None, interpolate=False, color='blue', 
    preset_ax=None, linestyle='-.', marker='o'):


    fig = plt.figure(figsize=(8, 8), linewidth=1.0)
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
    

    if z_values is None:
        main_ax = sns.kdeplot(x, y,
                 # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
                 cmap='Oranges',
                 shade=True, shade_lowest=True,
                 fontsize=10, ax=main_ax, linewidths=1,
                 vertical=True)
        # main_ax.legend(lbl, 
        #   loc='lower left', fontsize=18,
        #   bbox_to_anchor=(1.05, 1.05, ),  borderaxespad=0)
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

    if z_values is not None:
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_interpolate = griddata(np.array([x, y]).T, z_values, (grid_x, grid_y), method='cubic')
        main_ax.imshow(grid_interpolate.T, extent=(min(x),max(x),min(y),max(y)), origin='lower')

    plt.title(title, **title_font)


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



def process_name(input_name, main_dir):
    name = [k.replace(main_dir, "") for k in input_name]
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
    #   kde_kws={"color": "blue", "lw": 1},
    #   # shade=True,
    #   # hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
    #   vertical=False, norm_hist=True)
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
    #   kde_kws={"color": "blue", "lw": 1},
    #   # shade=True,
    #   # hist_kws={"linewidth": 3, "alpha": 0.3, "color": "blue"},
    #   vertical=True, norm_hist=True)
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


def scatter_plot_5(x, y, z_values=None, list_cdict=None, xvlines=None, yhlines=None, 
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

    # main_ax = sns.kdeplot(x, y,
    #        # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
    #        cmap='Oranges',
    #        shade=True, shade_lowest=False,
    #        fontsize=10, ax=main_ax, linewidths=1,
    #        # vertical=True
    #        )

    for _m, _cdict, _x, _y, _a in zip(marker, list_cdict, x, y, alphas):
        if _m == "+":
            main_ax.scatter(_x, _y, s=5, 
                marker=_m, c="black", 
                alpha=_a, edgecolor="black")
        else: 
            if len(_cdict.keys()) == 1:
                main_ax.scatter(_x, _y, s=s, 
                    marker=_m, c=list(_cdict.keys())[0], 
                    alpha=_a, edgecolor="black")
            else:
                plt_half_filled(ax=main_ax, x=_x, y=_y, 
                    cdict=_cdict, alpha=_a
                    )

    if z_values is not None:
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_interpolate = griddata(np.array([x, y]).T, z_values, (grid_x, grid_y), method='cubic')
        main_plot = main_ax.imshow(grid_interpolate.T, 
            extent=(min(x),max(x),min(y),max(y)), origin='lower',
            cmap="jet")

    fig.colorbar(main_plot, ax=main_ax)

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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    return fig.colorbar(mappable, cax=cax, shrink=0.6)

def myround(x, base=5):
    return base * round(x/base)

def scatter_plot_6(x, y, z_values=None, list_cdict=None, xvlines=None, yhlines=None, 
        sigma=None, mode='scatter', lbl=None, name=None, 
        s=100, alphas=0.8, title=None,
        x_label='x', y_label='y', 
        save_file=None, interpolate=False, color='blue', 
        preset_ax=None, linestyle='-.', marker='o',
        cmap='seismic',
        vmin=None, vmax=None
        ):

    org_x = copy.copy(x)
    org_y = copy.copy(y)
    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)

    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) * 200
    x = x.T[0]
    y = y.T[0]

    tick_pos = [0.0, 50.0, 100.0, 150.0, 200.0]
    tmp = np.arange(min_org_x, max_org_x, (max_org_x - min_org_x)/len(tick_pos))
    xticklabels = [myround(k,5) for k in tmp]

    tmp = np.arange(min_org_y, max_org_y, (max_org_y - min_org_y)/len(tick_pos))
    yticklabels = [myround(k,5) for k in tmp]


    fig, main_ax = plt.subplots(figsize=(10, 9), linewidth=1.0) # 
    # grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    if False:
        main_ax = sns.kdeplot(x, y,
             # joint_kws={"colors": "black", "cmap": None, "linewidths": 3.0},
             cmap='Oranges', # Greys
             shade=True, shade_lowest=False,
             fontsize=10, ax=main_ax, linewidths=1,
             alpha=0.5
             # vertical=True
             )


    sns.set_style(style='white') 
    for _m, _cdict, _x, _y, _a in zip(marker, list_cdict, x, y, alphas):
        if _m in ["o", "D", "*"]:
            main_ax.scatter(_x, _y, s=s, 
                marker=_m, c="white", 
                alpha=1.0, edgecolor="red")
        elif _m == ".":
            # # for unlbl cases
            main_ax.scatter(_x, _y, s=5, 
                marker=_m, c="black", 
                alpha=_a, edgecolor=None)
        else: 
            if len(_cdict.keys()) == 1:
                main_ax.scatter(_x, _y, s=s, 
                    marker=_m, c=list(_cdict.keys())[0], 
                    alpha=_a, edgecolor="black")
            else:
                plt_half_filled(ax=main_ax, x=_x, y=_y, 
                    cdict=_cdict, alpha=_a
                    )

    xx, yy = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # density
    # # # adddd
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Contourf plot
    # cs = main_ax.contour(xx, yy, f, 
    #       levels=3, corner_mask=False,
    #       extent=None, colors="gray",
    #       ) 


    # # z_value layer
    if True:
        if z_values is not None:
            grid_interpolate = griddata(np.array([x, y]).T, z_values, (xx, yy), 
                method='nearest')

            # max_ipl = 0.8*max([abs(np.nanmax(grid_interpolate.T)), abs(np.nanmin(grid_interpolate.T))])
            # max_ipl = 2.2

            # orig_cmap = mpl.cm.coolwarm
            # shrunk_cmap = shiftedColorMap(orig_cmap, 
            #   start=np.nanmin(grid_interpolate.T), 
            #   midpoint=0.5, stop=np.nanmax(grid_interpolate.T), name='shrunk')
            if vmin is None:
                vmin = np.nanmin(grid_interpolate.T)
            if vmax  is None:
                vmax = np.nanmax(grid_interpolate.T)

            # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            norm = colors.DivergingNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            z_plot = main_ax.imshow(grid_interpolate.T, 
                extent=(min(x),max(x),min(y),max(y)), origin='lower',
                cmap=cmap, norm=norm, 
                # vmin=-max_ipl, vmax=max_ipl, 
                interpolation="hamming",
                alpha=0.9)
            colorbar(z_plot)
            # if ".png" not in save_file:
            #   fig.colorbar(z_plot, shrink=0.6)


    # for xvline in xvlines:
    #   main_ax.axvline(x=xvline, linestyle='-.', color='black')
    # for yhline in yhlines:
    #   main_ax.axhline(y=yhline, linestyle='-.', color='black')
    # main_ax.set_title(title, **title_font)
    # main_ax.set_xlabel(x_label, **axis_font)
    # main_ax.set_ylabel(y_label, **axis_font)

    
    if name is not None:
        for i in range(len(x)):
            main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)
    plt.xticks(tick_pos, []) # xticklabels, size=14
    plt.yticks(tick_pos, []) # yticklabels, size=14

    main_ax.set_aspect('auto')
    
    plt.tight_layout(pad=1.1)
    makedirs(save_file)
    plt.savefig(save_file, transparent=False)
    print ("Save at: ", save_file)
    release_mem(fig=fig)

def scatter_plot_7(x, y, z_values=None, list_cdict=None, xvlines=None, yhlines=None, 
    sigma=None, mode='scatter', lbl=None, name=None, 
    s=100, alphas=0.8, title=None,
    x_label='x', y_label='y', 
    save_file=None, interpolate=False, color='blue', 
    preset_ax=None, linestyle='-.', marker='o',
    cmap='seismic',
    vmin=None, vmax=None
    ):

    org_x = copy.copy(x)
    org_y = copy.copy(y)
    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)

    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) * 200
    x = x.T[0]
    y = y.T[0]

    tick_pos = [0.0, 50.0, 100.0, 150.0, 200.0]
    tmp = np.arange(min_org_x, max_org_x, (max_org_x - min_org_x)/len(tick_pos))
    xticklabels = [myround(k,5) for k in tmp]

    tmp = np.arange(min_org_y, max_org_y, (max_org_y - min_org_y)/len(tick_pos))
    yticklabels = [myround(k,5) for k in tmp]
    # x = copy.copy(XY[:, 0]) 
    # y = copy.copy(XY[:, 1]) 
    # print (x)
    # print (len(x))

    fig, main_ax = plt.subplots(figsize=(9, 8), linewidth=1.0) # 
    # grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    sns.set_style(style='white') 

    if z_values is not None:
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
        grid_interpolate = griddata(np.array([x, y]).T, z_values, (grid_x, grid_y), 
            method='cubic')
        if vmin is None:
            vmin = np.nanmin(grid_interpolate.T)
        if vmax  is None:
            vmax = np.nanmax(grid_interpolate.T)

        # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        norm = colors.DivergingNorm(vmin=0.0, vcenter=vmax/4, vmax=vmax)
        # norm = colors.DivergingNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        z_plot = main_ax.imshow(grid_interpolate.T, 
            extent=(min(x),max(x),min(y),max(y)), origin='lower',
            cmap=cmap,
            norm=norm, 
            # vmin=-max_ipl, vmax=max_ipl, 
            interpolation="hamming",
            alpha=1.0)
        # colorbar(z_plot)
        fig.colorbar(z_plot, shrink=0.6)


    # for xvline in xvlines:
    #   main_ax.axvline(x=xvline, linestyle='-.', color='black')
    # for yhline in yhlines:
    #   main_ax.axhline(y=yhline, linestyle='-.', color='black')
    # main_ax.set_title(title, **title_font)
    # main_ax.set_xlabel(x_label, **axis_font)
    # main_ax.set_ylabel(y_label, **axis_font)

    plt.xticks(tick_pos, []) # xticklabels, size=14
    plt.yticks(tick_pos, []) # yticklabels, size=14

    if name is not None:
        for i in range(len(x)):
            main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

    main_ax.set_aspect('auto')
    
    plt.tight_layout(pad=1.1)
    makedirs(save_file)
    plt.savefig(save_file, transparent=False)
    print ("Save at: ", save_file)
    release_mem(fig=fig)



def scatter_plot_8(x, y, z_values=None, list_cdict=None, xvlines=None, yhlines=None, 
    sigma=None, mode='scatter', lbl=None, name=None, 
    s=100, alphas=0.8, title=None,
    x_label='x', y_label='y', 
    save_file=None, interpolate=False, color='blue', 
    preset_ax=None, linestyle='-.', marker='o',
    cmap='seismic',
    vmin=None, vmax=None
    ):

    org_x = copy.copy(x)
    org_y = copy.copy(y)

    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)

    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) * 200
    x = x.T[0]
    y = y.T[0]

    tick_pos = [0.0, 50.0, 100.0, 150.0, 200.0]
    tmp = np.arange(min_org_x, max_org_x, (max_org_x - min_org_x)/len(tick_pos))
    xticklabels = [myround(k,5) for k in tmp]

    tmp = np.arange(min_org_y, max_org_y, (max_org_y - min_org_y)/len(tick_pos))
    yticklabels = [myround(k,5) for k in tmp]


    fig, main_ax = plt.subplots(figsize=(8, 8), linewidth=1.0) # 
    # grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    sns.set_style(style='white') 

    for _m, _cdict, _x, _y, _a in zip(marker, list_cdict, x, y, alphas):
        # if _z == 0:
        #   continue
        if _m in ["o", "D", "*"]:
            main_ax.scatter(_x, _y, s=s, 
                marker=_m, c="white", 
                alpha=1.0, edgecolor="red")
        elif _m == ".":
            # # for unlbl cases
            main_ax.scatter(_x, _y, s=5, 
                marker=_m, c="black", 
                alpha=_a, edgecolor=None)
        else: 
            if len(_cdict.keys()) == 1:
                main_ax.scatter(_x, _y, s=s, 
                    marker=_m, c=list(_cdict.keys())[0], 
                    alpha=_a, edgecolor="black")
            else:
                plt_half_filled(ax=main_ax, x=_x, y=_y, 
                    cdict=_cdict, alpha=_a
                    )
    xx, yy = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Contourf plot
    cs = main_ax.contour(xx, yy, f, 
            levels=3, corner_mask=False,
            extent=None, colors="gray",
            ) 

    # if z_values is not None:
    #   ft_ids = np.where(z_values!=0)[0]
    #   xf = x[ft_ids]
    #   yf = y[ft_ids]
    #   xmin, xmax = min(x), max(x)
    #   ymin, ymax = min(y), max(y)

    #   # # estimate only xf, yf
    #   xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #   positions = np.vstack([xx.ravel(), yy.ravel()])
    #   values = np.vstack([xf, yf])
    #   kernel = stats.gaussian_kde(values)
    #   f = np.reshape(kernel(positions).T, xx.shape)

    #   main_ax.set_xlim(xmin, xmax)
    #   main_ax.set_ylim(ymin, ymax)
    #   # Contourf plot
    #   cfset = main_ax.contourf(xx, yy, f, cmap='Blues')

    # for xvline in xvlines:
    #   main_ax.axvline(x=xvline, linestyle='-.', color='black')
    # for yhline in yhlines:
    #   main_ax.axhline(y=yhline, linestyle='-.', color='black')
    # main_ax.set_title(title, **title_font)
    # main_ax.set_xlabel(x_label, **axis_font)
    # main_ax.set_ylabel(y_label, **axis_font)

    plt.xticks(tick_pos, []) # xticklabels, size=14
    plt.yticks(tick_pos, []) # yticklabels, size=14

    if name is not None:
        for i in range(len(x)):
            main_ax.annotate(name[i], xy=(x[i], y[i]), size=size_text)

    main_ax.set_aspect('auto')
    
    plt.tight_layout(pad=1.1)
    makedirs(save_file)
    plt.savefig(save_file, transparent=False)
    print ("Save at: ", save_file)
    release_mem(fig=fig)


def comp_kde2d(x, y, z_values=None, list_cdict=None, xvlines=None, yhlines=None, 
    sigma=None, mode='scatter', lbl=None, name=None, 
    s=100, alphas=0.8, title=None,
    x_label='x', y_label='y', 
    save_file=None, interpolate=False, color='blue', 
    preset_ax=None, linestyle='-.', marker='o',
    cmap='seismic',
    vmin=None, vmax=None
    ):

    # grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    sns.set_style(style='white') 

    plt.xticks(tick_pos, []) # xticklabels, size=14
    plt.yticks(tick_pos, []) # yticklabels, size=14



def fts_on_embedding(term, pv, estimator, X_train, y_train,
                ref_ids,
                X_all, xy, savedir, background, 
                vmin=None, vmax=None, cmap="jet"):
    # fig = plt.subplots(nrows=1,  sharey=True)
    fig = plt.figure(figsize=(9, 9))    
    # norm = mpl.colors.Normalize(vmin=0, vmax=20) # 
    # cmap = cm.jet # gist_earth
    # m = cm.ScalarMappable(norm=norm, cmap=cmap)
    c_dict = dict({
            "s1":"darkblue", "s2":"green",
            "p1":"purple", "d2":"cyan", "d5":"red", 
            "d10":"darkgreen", "d6":"gray", "d7":"brown", "f6":"orange",
            })
    # # # cyan, blue, 

    # # # setting fig only
    fig, main_ax = plt.subplots(figsize=(8, 8), linewidth=1.0) # 
    org_x = copy.copy(xy[:, 0])
    org_y = copy.copy(xy[:, 1])
    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)
    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) * 200
    x = x.T[0]
    y = y.T[0]
    tick_pos = [0.0, 50.0, 100.0, 150.0, 200.0]
    tmp = np.arange(min_org_x, max_org_x, (max_org_x - min_org_x)/len(tick_pos))
    xticklabels = [myround(k,5) for k in tmp]
    tmp = np.arange(min_org_y, max_org_y, (max_org_y - min_org_y)/len(tick_pos))
    yticklabels = [myround(k,5) for k in tmp]
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    main_ax.set_xlim(xmin, xmax)
    main_ax.set_ylim(ymin, ymax)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    # # # end setting fig only


    # # show y_obs
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_interpolate = griddata(np.array([x, y]).T, background, (grid_x, grid_y), 
        method='nearest')
    if vmin is None:
            vmin = np.nanmin(grid_interpolate.T)
    if vmax  is None:
        vmax = np.nanmax(grid_interpolate.T)
    # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    norm = colors.DivergingNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    z_plot = main_ax.imshow(grid_interpolate.T, 
        extent=(min(x),max(x),min(y),max(y)), origin='lower',
        cmap=cmap,
        norm=norm, 
        # vmin=-max_ipl, vmax=max_ipl, 
        interpolation="hamming",
        alpha=0.9)

    # # prepare condition of bkg
    values_ref = np.vstack([x[ref_ids], y[ref_ids]])
    kernel_ref = stats.gaussian_kde(values_ref)
    f_ref = np.reshape(kernel_ref(positions).T, xx.shape)
    cs = main_ax.contour(xx, yy, f_ref, 
            levels=1, corner_mask=False,
            extent=None, colors="black", linestyle="-."
            ) 
    fmt = {}
    for l in cs.levels:
        fmt[l] = "target"
    main_ax.clabel(cs, inline=1, fontsize=10, fmt=fmt)

    # # # end show y_obs
    save_file = savedir + "{0}.pdf".format(term)
    idp_test = dict()
    for i, v in enumerate(pv):
        z_values = X_all[:, i]

        if len(set(z_values)) >1 and term in v:
            ft_ids = np.where(z_values!=0)[0]
            print (len(ft_ids))
            xf = x[ft_ids]
            yf = y[ft_ids]

            if "of" in v:
                c_term = v[:v.find("-")]
            else:
                c_term = v[v.find("-")+1:]
            values = np.vstack([xf, yf])
            kernel = stats.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)

            BC_coeff = np.sum(np.sqrt(f * f_ref))
            print (term, c_term, BC_coeff)
            idp_test[v] = BC_coeff
            # Contourf plot
            levels = 1
            fmt = {}
            cs = main_ax.contour(xx, yy, f, 
                    levels=levels, corner_mask=False,
                    extent=None, colors=c_dict[c_term],
                    label=v
                    ) 
            for l in cs.levels:
                fmt[l] = v
            main_ax.clabel(cs, inline=1, fontsize=10, fmt=fmt)

            cs.cmap.set_under('white')

            # for j, a in enumerate(main_ax.flatten()):
            #   # if j == 0:
            #   #   a.set_ylabel(v, rotation=0)
            #   # else:
            #   #   a.set_ylabel("")
            #   if j == 1:
            #       a.set_ylabel("")
            #   else:
            #       a.get_legend().remove()
            #   a.set_ylim([-0.1, 0.7])
                # a.set_xlabel("")
                # a.set_xticklabels([])
                # a.set_yticklabels([])
            # plt.ylabel(str(v))
    
    main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(tick_pos, []) # xticklabels, size=14
    plt.yticks(tick_pos, []) # yticklabels, size=14
    
    main_ax.set_aspect('auto')
    plt.tight_layout(pad=1.1)
    makedirs(save_file)
    print ("Save at: ", save_file)
    plt.savefig(save_file, transparent=False, bbox_inches="tight")

    release_mem(fig=fig)
    return idp_test




def dump_interpolate(x, y, z_values=None, 
                save_file=None):

    org_x = copy.copy(x)
    org_y = copy.copy(y)
    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)

    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) * 200
    x = x.T[0]
    y = y.T[0]

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_interpolate = griddata(np.array([x, y]).T, z_values, (grid_x, grid_y), method='cubic')

    makedirs(save_file)
    np.savetxt(save_file, grid_interpolate)

def imshow(grid, cmap, save_file, vmin=None, vmax=None):
    fig, main_ax = plt.subplots(figsize=(8, 8), linewidth=1.0) # 

    if vmin is None:
        vmin = np.nanmin(grid.T)
    if vmax  is None:
        vmax = np.nanmax(grid.T)

    # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    norm = colors.DivergingNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    z_plot = main_ax.imshow(grid.T, 
        cmap=cmap,
        norm=norm, 
        # vmin=-max_ipl, vmax=max_ipl, 
        interpolation="hamming",
        alpha=0.8)
    # colorbar(z_plot)
    fig.colorbar(z_plot, shrink=0.6)
    plt.tight_layout(pad=1.1)
    makedirs(save_file)
    plt.savefig(save_file, transparent=False)
    print ("Save at: ", save_file)
    release_mem(fig=fig)

def gradient_map(list_vectors, save_file):
    fig, ax = plt.subplots(figsize=(8, 8), linewidth=1.0)
    # ax1.set_title()
    for y, rows in enumerate(list_vectors):
        for x, column_point in enumerate(rows):
            if not np.isnan(column_point[0]):
                Q = ax.quiver(x, y, column_point[0], column_point[1], 
                    units='x',# width=0.001
                    scale_units="inches", scale=10, headwidth=0.5, headlength=0.5
                    )
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
    #   horizontalalignment='center', size=14, 
    #   color=color, weight='semibold')
    patch = bplot['boxes'][0]
    patch.set_facecolor(color)

    # # midle axis
    if is_shown_tails:
        y_star_ax.scatter([pos_x], [y_min], s=100, marker="+", 
            c=color, alpha=1.0, edgecolor="black")

        # bplot = y_star_ax.boxplot(x=y, vert=True, #notch=True, 
        #       # sym='rs', # whiskerprops={'linewidth':2},
        #       positions=[pos_x], patch_artist=True,
        #       widths=0.1, meanline=True, #flierprops=flierprops,
        #       showfliers=True, showbox=True, showmeans=True
        #       )
        # y_star_ax.text(pos_x, y_min, round(y_min, 2),
        #   horizontalalignment='center', size=14, 
        #   color=color, weight='semibold')
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
    ax = df.plot.bar(x='lab', y='val', rot=0)

    # plt.bar(xs, hist,  alpha=1.0)
    # y_plot = hist
    # y_plot, x_plot, patches = ax.hist(x, bins=nbins, histtype='stepfilled', # step, stepfilled, 'bar', 'barstacked'
    #                                   density=True, label=label, log=False,  
    #                                   color='black', #edgecolor='none',
    #                                   alpha=1.0, linewidth=2)

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

def two_surf(surf1, surf2, lbl1, lbl2, save_at, title):
    x1 = range(surf1.shape[0])
    y1 = range(surf1.shape[1])
    x1, y1 = np.meshgrid(x1, y1)


    x2 = range(surf2.shape[0])
    y2 = range(surf2.shape[1])
    x2, y2 = np.meshgrid(x2, y2)
    
    lift = 2
    surf2 += lift
    vmin, vmax = np.nanmin(surf2), np.nanmax(surf2)

    fig = plt.figure(figsize=(8, 8))

    ax = Axes3D(fig)
    s1 = ax.plot_surface(x1, y1, surf1, alpha=1.0, 
        cmap="PiYG", # cmap=cm.hot, #color="orange", 
        shade=False, #rcount=500, ccount=500, 
        linewidth=0.1, linestyle="-", antialiased=True)
    # surf = ax.plot_trisurf(xi, yi, zi, alpha=0.2, cmap="jet", 
    #   linewidth=0.1, vmin=min(zi.ravel()), vmax=max(zi.ravel()))

    s2 = ax.plot_surface(x2, y2, surf2, alpha=1.0, 
        cmap="Blues_r", # cmap=cm.hot, #color="orange", 
        shade=False, #rcount=500, ccount=500, 
        linewidth=0.1, linestyle="-", antialiased=True,
        zorder=3, vmin=lift, vmax=vmax*0.8)

    # for surf in (surf1, surf2):
    #   surf._facecolors2d = surf._facecolors3d
    #   surf._edgecolors2d = surf._edgecolors3d

    # ax.set_axis_off()
    ax.view_init(20, 30) # 60, 60, 37, -150
    # plt.show()
    plt.title(title, **title_font)
    ax_setting_3d(ax=ax)
    fig.patch.set_visible(False)
    fig.colorbar(s1, shrink=0.4, label=lbl1)
    fig.colorbar(s2, shrink=0.4, label=lbl2)


    plt.tight_layout(pad=1.1)

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
        #   linewidth=0.1, vmin=min(zi.ravel()), vmax=max(zi.ravel()))

        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_axis_off()
        ax.view_init(64, -147) # 60, 60, 37, -150
        ax_setting_3d(ax=ax)
        fig.patch.set_visible(False)

    if mode == "3D_patch":
        ax = fig.add_subplot(1, 1, 1, projection='3d')
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

def plt_half_filled(ax, x, y, cdict, alpha):

    # small_ratio, big_ratio = sorted(cdict.values())
    # small_color, big_color = sorted(cdict, key=cdict.get)
    print (cdict)
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

    HalfA = mpl.patches.Wedge((x, y), 0.9, alpha=alpha, 
        theta1=0-rot,theta2=angle1-rot, facecolor=color1, 
        lw=0.5,
        edgecolor="black")
    HalfB = mpl.patches.Wedge((x, y), 1.8, alpha=alpha,
        theta1=angle1-rot,theta2=360-rot, facecolor=color2,
        lw=0.5,
        edgecolor="black")

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
    #   theta1=0-rot,theta2=ratio-rot, color='r')
    # HalfB = mpl.patches.Wedge((i, i), 5,
    #   theta1=ratio-rot,theta2=360-rot, color='b')
    # # rot=rot+360/len(x)

    # ax.add_artist(HalfA)
    # ax.add_artist(HalfB)

    # ax.set_xlim((-10, 10))
    # ax.set_ylim((-10, 10))
    # plt.savefig("test_half_filled.pdf")
    index =  "mix/Sm-Fe10-Al1-Ga1"
    # colors = get_color_112(index) 
    r = get_ratio(index=index, element="Ga")

def plot_heatmap(matrix, vmin, vmax, save_file, cmap, lines=None, title=None):
    if vmax is None:
        vmax = np.max(np.array(matrix))
    if vmin is None:
        vmin = np.min(np.array(matrix))
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    print (matrix)
    print (vmin, vmax)

    ax = sns.heatmap(matrix, cmap=cmap, 
            xticklabels=True,
            yticklabels=True,
            annot_kws={"size":1},
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


def plot_ppde(term, pv, estimator, X_train, y_train,
                X_all, xy, savedir):
    # fig = plt.subplots(nrows=1,  sharey=True)
    fig = plt.figure(figsize=(16, 8))   
    norm = mpl.colors.Normalize(vmin=0, vmax=20) # 
    cmap = cm.jet # gist_earth
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    c_list = ["black", "blue", "green", "purple", "brown", "gray", "black", "cyan", "lime",
    "darkgreen", "navy", "magenta"]
    # i = 0
    # for v, ax in zip(pv, axes.ravel()):

    estimator = estimator.fit(X_train, y_train)
    display = plot_partial_dependence(estimator, X_train, [0, 1],
        kind='average', n_jobs=3, grid_resolution=20,
        line_kw=dict({"color": "red", "label":"formation energy", "linestyle":"-."}))
    ax = display.axes_

    for i, v in enumerate(pv):
        z_values = X_all[:, i]

        if len(set(z_values)) >1 and term in v:
            # try:
                # scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
                #   z_values=z_values,
                #   list_cdict=list_cdict, 
                #   xvlines=[0.0], yhlines=[0.0], 
                #   sigma=None, mode='scatter', lbl=None, name=None, 
                #   s=60, alphas=alphas, 
                #   title=save_file.replace(ALdir, ""),
                #   x_label=FLAGS.embedding_method + "_dim_1",
                #   y_label=FLAGS.embedding_method + "_dim_2", 
                #   interpolate=False, cmap="seismic",
                #   save_file=save_file,
                #   preset_ax=None, linestyle='-.', marker=marker_array,
                #   vmin=None, vmax=None
                #   )
            save_file= savedir + "{0}.pdf".format(term)
            tmp_estimator = copy.copy(estimator)
            tmp_estimator = tmp_estimator.fit(xy, z_values)

            display = plot_partial_dependence(tmp_estimator, xy, [0, 1],
                   kind='average', n_jobs=3, grid_resolution=20,
                    ax=ax, line_kw=dict({"color": c_list[i % (len(c_list))], "label":v})) # m.to_rgba(i)
            ax = display.axes_
            f = display.figure_
            # f.suptitle(v)
            for j, a in enumerate(ax.flatten()):
                # if j == 0:
                #   a.set_ylabel(v, rotation=0)
                # else:
                #   a.set_ylabel("")
                if j == 1:
                    a.set_ylabel("")
                    a.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    a.get_legend().remove()
                a.set_ylim([-0.1, 0.7])
                # a.set_xlabel("")
                # a.set_xticklabels([])
                # a.set_yticklabels([])
            # plt.ylabel(str(v))
            # plt.legend()

            print (str(v))
            makedirs(save_file)
            plt.savefig(save_file, transparent=False, bbox_inches="tight")

                # except:
                #   pass
                # i += 1
                # if i == n_plots:
                #   break
    release_mem(fig=fig)



def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 2)
    ax = plt.gca()
    font_size = abs(corr_r) * 80 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=70)


def pairplot(df, fix_cols, term, save_dir):
    fig = plt.figure(figsize=(8, 8))    

    all_cols = df.columns
    interested_cols = [v for v in all_cols if term in v]
    interested_cols = np.concatenate((interested_cols, fix_cols))

    print (interested_cols)

    plot_df = df[interested_cols]
    save_at = save_dir + "/{}.pdf".format(term)

    sns.set(style='white', font_scale=1.6)
    # g = sns.PairGrid(plot_df, aspect=1.5, diag_sharey=False, despine=False)
    # sns.set_theme(style="ticks")

    g = sns.pairplot(plot_df, plot_kws={"s": 20, "color":"black"})

    # g.map_lower(sns.regplot, lowess=True, ci=False,
    #             line_kws={'color': 'red', 'lw': 3},
    #             scatter_kws={'color': 'black', 's': 5})
    # g.map_diag(sns.distplot, color='black',
    #            kde_kws={'color': 'red', 'cut': 0.7, 'lw': 1},
    #            hist_kws={'histtype': 'bar', 'lw': 2,
    #                      'edgecolor': 'k', 'facecolor':'grey'})
    # g.map_diag(sns.rugplot, color='black')
    # g.map_upper(corrdot)
    # g.map_upper(corrfunc)
    # g.fig.subplots_adjust(wspace=0, hspace=0)

    # Remove axis labels
    # for ax in g.axes.flatten():
    #     ax.set_ylabel('')
    #     ax.set_xlabel('')

    # # Add titles to the diagonal axes/subplots
    # for ax, col in zip(np.diag(g.axes), iris.columns):
    #     ax.set_title(col, y=0.82, fontsize=26)

    makedirs(save_at)
    plt.savefig(save_at)
    release_mem(fig)

def get_2d_interpolate(x, y, z_values):
    org_x = copy.copy(x)
    org_y = copy.copy(y)

    min_org_x, max_org_x = min(org_x), max(org_x)
    min_org_y, max_org_y = min(org_y), max(org_y)

    x = MinMaxScaler().fit_transform(np.array(org_x).reshape(-1, 1)) # * 200
    y = MinMaxScaler().fit_transform(np.array(org_y).reshape(-1, 1)) # * 200
    x = x.T[0]
    y = y.T[0]
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

    grid_interpolate = griddata(np.array([x, y]).T, z_values, (grid_x, grid_y), 
            method='nearest')
    grid_interpolate = grid_interpolate.T

    return grid_interpolate


def curve_fit(x, y, ax, label, marker, c, is_fit=False):
    gp_kernel = ConstantKernel(constant_value=1)*RBF(length_scale=0.3) + WhiteKernel(noise_level=0.01)
    reg = GaussianProcessRegressor(kernel=gp_kernel)

    # reg = utils.get_model("u_gp", 1, True, n_shuffle=10000,
    #         mt_kernel=None)

    X_train = MinMaxScaler().fit_transform(x)
    reg.fit(X_train, y)
    # ymean = reg.predict(X_train)
    # ystd = reg.predict_proba(X_train)
    ymean, ystd = reg.predict(X_train, return_std=True)

    ax.scatter(x, y, color=c, marker=marker, label=label)
    # ax.plot(x, ymean, color=c)

    # # try:
    # ax.fill_between(x.ravel(), ymean-ystd/2, ymean+ystd/2,
    #     color=c, alpha=0.2,
    #     )
    # except Exception as e:
    #     pass
    return ax


if __name__ == "__main__":
    test_half_filled()









