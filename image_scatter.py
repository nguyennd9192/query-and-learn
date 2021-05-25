import numpy as np
import pandas as pd
import seaborn as sns
import time, gc, os
import cv2 as cv
import pickle
import warnings

from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, minmax_scale

warnings.filterwarnings("ignore") 

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()

def makedirs(file):
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

def coords_file_gen(gen_dir):
    save_file = "{0}/coords.csv".format(gen_dir)
    if not os.path.isfile(save_file):
        x_df = pd.read_csv('{0}/x.txt'.format(gen_dir), sep=" ", header=None)
        x_df.columns = ["x"]
        y_df = pd.read_csv('{0}/y.txt'.format(gen_dir), sep=" ", header=None)
        y_df.columns = ["y"]
        coords_df = pd.concat([x_df, y_df], axis=1)

        img_dir = "{0}/particles_image_2D_filter/".format(gen_dir)
        img_files = [f for f in os.listdir(img_dir) if f[-4:]==".jpg"]
        srt_img_files = sorted(img_files, key= lambda x: int(x.split("_")[1]), reverse=False)
        coords_df["file"] = srt_img_files
        coords_df.to_csv("{0}/coords.csv".format(gen_dir), index=False)
    else:
        print("Coordinate file is available")
        coords_df = pd.read_csv(save_file)
    return coords_df


def images_scatter(coords_df, 
    input_dir="./Visualize/", n_bins=50, 
    margin=0.1, save_fig=False, more_info=None):
    coords = coords_df[["x", "y"]].values
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(coords)
    save_fig_dir = "{0}/figures/".format(input_dir)
    makedirs(save_fig_dir)
    
    # Divide the images into bins 
    bins = np.linspace(0, 1, n_bins+1)
    ins_x = np.digitize(coords[:, 0], bins)
    ins_x[ins_x>n_bins] = n_bins
    ins_y = np.digitize(coords[:, 1], bins)
    ins_y[ins_y>n_bins] = n_bins

    grid_txt = ["{0}|{1}".format(ins_yi, ins_xi) for ins_xi, ins_yi in zip(ins_x, ins_y)]
    grid_size= 1/n_bins
    n_col = n_bins + 2*int(margin/grid_size)
    n_row = n_bins + 2*int(margin/grid_size)

    fig = plt.figure(figsize=(15, 15))
    grid = plt.GridSpec(n_row, n_col)
    grid.update(wspace=0., hspace=0.) 
    plt.rcParams["axes.linewidth"] = 1
    # Plot on the background 
    sns.kdeplot(coords[:, 0], coords[:, 1], shade=True, cmap="Oranges")
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0-margin, 1+margin])
    plt.ylim([0-margin, 1+margin])
    plt.box(False)
    
    # Show the image representing in each bin
    for gt in np.unique(grid_txt):
        check_idxes = [i for i in np.arange(len(grid_txt)) if grid_txt[i]==gt]
        sub_df = coords_df.iloc[check_idxes]
        check_coords = coords[check_idxes]
        ref_coords = np.mean(check_coords, axis=0)
        check_img_idx = np.argmin(np.mean(abs(check_coords-ref_coords)))
        check_img_file = sub_df.iloc[check_img_idx]["file"]

        grid_coords = np.array(gt.split("|")).astype(np.int32) - 1 + int(margin/grid_size)
        # print(check_img_file, grid_coords)
        ax = fig.add_subplot(grid[n_row-grid_coords[0], grid_coords[1]])
        ax.margins(0, 0)
        try:
            img = plt.imread("{0}/{1}".format(input_dir, check_img_file))

            ax.imshow(img)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.autoscale_view('tight')
        except Exception as e:
            pass


    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if save_fig: 
        if more_info is not None:
            save_file = "{0}/img_map_nbins{1}_{2}.pdf".format(save_fig_dir, n_bins, more_info)
        else:
            save_file = "{0}/img_map_nbins{1}.pdf".format(save_fig_dir, n_bins)
        plt.savefig(save_file, transparent=True, dpi=1000)
        print('Save at: {}'.format(save_file))
        release_mem(fig)
    else: 
        plt.show()

if __name__ == "__main__":
    # n_bins as the number of grid applied on the plot to sample images
    # increase n_bins for denser scatter of images
    for idx in np.arange(1, 5):
        print("Sample {}".format(idx))
        source_dir = "/Users/anh_1/Downloads/sample_{}/".format(idx)
        coords_df = coords_file_gen(source_dir)
        print(coords_df.head(5))

        start = time.time()
        images_scatter(coords_df, input_dir=source_dir, n_bins=50, margin=0.1, save_fig=True, more_info=None)
        end = time.time()
        print("Finish in: {}".format(end-start))
        print("====================================================")
