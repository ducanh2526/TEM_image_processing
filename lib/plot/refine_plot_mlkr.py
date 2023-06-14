import os
import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import tsem
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from scipy.interpolate import Rbf

lib_dir = "../lib/"
if not lib_dir in sys.path:
    sys.path.append(lib_dir)
from general_lib import *

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

markers = [".", "^",  "o", "+", "."]
# dts_colors = ["black", "orange", "blue", "red", "lime"]
dts_colors = np.array([[0.36078431, 0.32941176, 0.31764706],
                        [0.55294118, 0.39215686, 0.30588235],
                        [0.7254902 , 0.4627451 , 0.27058824],
                        [0.8745098 , 0.54901961, 0.2       ],
                        [1.        , 0.65098039, 0.        ]])[::-1]

variable_signs = {'d_min': r'$\mathsf{r}_{min}$', 'D_min_alg': r'$\mathsf{d}^{aligned}_{min}$', 
                 'D_min': r'$\mathsf{d}_{min}$','mean_D': r'$\bar{\mathsf{d}}$', 'var_D': r'$\sigma_{\mathsf{d}}^{2}$', 
                 'F_D': r'$\mathsf{F_{d}}$', 'A_D': r'$\mathsf{A}^{front}$',
                 'X_base_l': r'$\delta_{x}$', 
                 'Y_base_l': r'$\delta_{y}$', 
                 'A_gs': r'$V$', 'mean_gs': r'$\overline{\mathsf{A}^{side}}$', 'var_gs': r'$\sigma^{2}_{\mathsf{A}}$', 
                 'F_gs': r'$\mathsf{F_{A}}$',
                 'k': r'$\mathsf{k}$', 'G': r'$\mathsf{G}$'}
unit_dict = {"d_min": "nm", "D_min": "nm", "mean_D": "nm", "X_base_l":"nm", "Y_base_l":"nm", 
            "var_D":r"nm$^2$", "F_D":r"nm$^{-1}$", "G":r"G$_0$", "k":"N/m"}
colors = ['forestgreen', 'darkgreen', 'seagreen', 'green', 'mediumseagreen', 
        'y', 'darkseagreen', 'olivedrab', 'olive', 'darkolivegreen', 'darkkhaki', 
        'goldenrod', 'darkgoldenrod', 'deeppink', 'red', 'gray']
feature_colors = {}
for i_f, f in enumerate(variable_signs.keys()):
    feature_colors[f] = colors[i_f]

all_features_df = pd.read_csv("../output/exp_data/All_datasets_features.csv", index_col=0)
selected_pairs = {"pull":[["3_41", "3_55"], ["2_26", "2_39"], ["1_523", "1_529"], ["1_37", "1_43"]], 
                "push": [["3_17", "3_25"], ["1_891", "1_896"], ["1_517", "1_523"], ["1_43", "1_49"]]}

def mlkr_scatter_pair(coords_df, selected_pairs, bck_var=None, marker_set=markers, color_set=colors, margin=0.1, 
                    sct_lw=2, sct_size=50, sct_alpha=1.0, legends=None,
                    more_info=None, save_fig=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # coords_df includes "x", "y", "label" and var for background plot 
    if "label" in coords_df.columns:
        assert len(marker_set)>=len(set(coords_df["label"]))
        assert len(color_set)>=len(set(coords_df["label"]))
        if legends is not None:
            assert len(legends)==len(set(coords_df["label"]))
    else:
        coords_df["label"] = [0 for rep in range(len(coords_df))]
    coords = coords_df[["x", "y"]].values
    coords = MinMaxScaler().fit_transform(coords)
    if bck_var is not None:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords_df[bck_var].values
        f = Rbf(x, y, z, kind="gaussian", smooth=5, episilon=5)
        x_grid = np.linspace(-margin, 1+margin, 200)
        y_grid = np.linspace(-margin, 1+margin, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y) 
        plt.pcolormesh(X, Y, Z, alpha=0.8, vmin=0, vmax=170, cmap=plt.cm.YlGn, shading="gouraud")

        ax.scatter(*coords_df[["x", "y"]].values.T, s=37.5, alpha=0.5, color="black", marker=".",
                    edgecolor=None, linewidths=0.3, zorder=5)

        for fr_pair in selected_pairs["push"]: 
            ax.scatter(coords_df.loc[fr_pair, "x"].values.T, coords_df.loc[fr_pair, "y"].values.T, s=sct_size*1.75, alpha=sct_alpha, facecolor="None", marker=marker_set[0], 
                        edgecolor="orange", linewidths=1.9, zorder=15)
            ax.plot(coords_df.loc[fr_pair, "x"].values.T, coords_df.loc[fr_pair, "y"].values.T, linestyle="dashed", lw=1.5, color=color_set[0], zorder=16)
            ax.arrow(coords_df.loc[fr_pair[0], "x"], coords_df.loc[fr_pair[0], "y"], 
                    coords_df.loc[fr_pair[1], "x"]-coords_df.loc[fr_pair[0], "x"], coords_df.loc[fr_pair[1], "y"]-coords_df.loc[fr_pair[0], "y"], 
                    color="black", linestyle="-", lw=1, zorder=16, head_width=0.025, head_length=0.03)
        
        for fr_pair in selected_pairs["pull"]: 
            ax.scatter(coords_df.loc[fr_pair, "x"].values.T, coords_df.loc[fr_pair, "y"].values.T, s=sct_size*1.75, alpha=sct_alpha, facecolor="None", marker=marker_set[1], 
                        edgecolor="orange", linewidths=1.9, zorder=12)
            ax.plot(coords_df.loc[fr_pair, "x"].values.T, coords_df.loc[fr_pair, "y"].values.T, linestyle="dashed", lw=1.5, color=color_set[1], zorder=16)
            ax.arrow(coords_df.loc[fr_pair[0], "x"], coords_df.loc[fr_pair[0], "y"], 
                    coords_df.loc[fr_pair[1], "x"]-coords_df.loc[fr_pair[0], "x"], coords_df.loc[fr_pair[1], "y"]-coords_df.loc[fr_pair[0], "y"], 
                    color="black", linestyle="-", lw=1, zorder=16, head_width=0.025, head_length=0.03)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0-margin, 1+margin])
        ax.set_ylim([0-margin, 1+margin])  
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.) 
        if bck_var is not None:
            # plt.box(False)
            fig.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.0, hspace=0.0)
        else:
            plt.tight_layout()
        if not save_fig is None:
            if more_info is not None:
                save_fig = "{0}_{1}.pdf".format(save_fig.split("."), more_info)
            plt.savefig(save_fig)
        else:
            plt.show()

def mlkr_cnt(trans_df, top_cnt=0.1, cnt_bw=0.05, margin=0.1, w_annot=False, save_fig=None):
    cnt_fig = plt.figure(figsize=(6, 6))
    ax = cnt_fig.add_subplot(111)
    check_features = list(feature_colors.keys()).copy()
    for i_f, feature in enumerate(check_features): 
        z = trans_df[feature].values
        if top_cnt>0:
            srt_idx = np.argsort(z)[::-1][:int(len(z)*top_cnt)]
        else:
            srt_idx = np.argsort(z)[:int(len(z)*-top_cnt)]
        x = trans_df["x"].values[srt_idx]
        y = trans_df["y"].values[srt_idx]
        x_grid = np.linspace(-margin, 1+margin, 200)
        y_grid = np.linspace(-margin, 1+margin, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        # f = Rbf(x, y, z[srt_idx], kind="gaussian", smooth=5, episilon=5)
        f = KernelDensity(kernel="gaussian", bandwidth=cnt_bw).fit(np.vstack([y.ravel(), x.ravel()]).T)
        Z = f.score_samples(np.vstack([Y.ravel(), X.ravel()]).T).reshape(X.shape)
        # z_cnt = sorted(z)[::-1][int(len(z)/4)-1]
        f_alpha = 1. if feature in ["k", "G"] else 0.6
        cs = ax.contour(X, Y, Z, zorder=int(i_f)+1, colors=feature_colors[feature], alpha=f_alpha, levels=[1.], linestyles="solid", extend="min", linewidth=3.)      
        if w_annot:  
            ax.clabel(cs, inline=True, fmt=variable_signs[feature], fontsize=18, colors=feature_colors[feature])
    ax.scatter(*trans_df[["x", "y"]].values.T, s=37.5, alpha=0.5, color="black", marker=".",
                edgecolor=None, linewidths=0.3, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0-margin, 1+margin])
    ax.set_ylim([0-margin, 1+margin])   
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.)
    # plt.box(False)
    # cnt_fig.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
    cnt_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.0)
    if save_fig is not None:
        if w_annot:
            save_fig = save_fig.split(".pdf")[0]+"_wAnnot.pdf"
        plt.savefig(save_fig)
        print("Save at: {}".format(save_fig))
    else: 
        plt.show()
    # release_mem(cnt_fig)

def mlkr_gen(coords_df, gen_coords_df, bck_var="k", margin=0.1, save_fig=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    coords = coords_df[["x", "y"]].values
    coords = MinMaxScaler().fit_transform(coords)
    if bck_var is not None:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords_df[bck_var].values
        f = Rbf(x, y, z, kind="gaussian", smooth=5, episilon=5)
        x_grid = np.linspace(-margin, 1+margin, 200)
        y_grid = np.linspace(-margin, 1+margin, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y) 
        plt.pcolormesh(X, Y, Z, alpha=0.8, vmin=0, vmax=170, cmap=plt.cm.YlGn, shading="gouraud")
    ax.scatter(*coords_df[["x", "y"]].values.T, s=37.5, alpha=0.5, color="black", marker="o",
                edgecolor="white", linewidths=0.3, zorder=10)
    ax.scatter(*gen_coords_df[["x", "y"]].values.T, s=15, alpha=0.3, color="red", marker="o",
                edgecolor="white", linewidths=0.2, zorder=5)
    ax.set_xlim([0-margin, 1+margin])
    ax.set_ylim([0-margin, 1+margin])  
    ax.set_xticks([])
    ax.set_yticks([])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.) 
    if bck_var is not None:
        fig.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.0, hspace=0.0)
    else:
        plt.tight_layout()
    if not save_fig is None:
        plt.savefig(save_fig)
    else:
        plt.show()

def images_scatter(coords_df, input_dir="./Visualize/", check_inst=None, n_bins=50, margin=0.1, bck_var="k", 
                    save_fig_dir=None, w_annot=False, more_info=None):
    # input DataFrame includes "x", "y" for coordinates and "file" for names of images
    scaler = MinMaxScaler()
    scaler.fit(coords_df[["x", "y"]].values)
    sub_coords_df = coords_df.loc[check_inst] if check_inst is not None else coords_df.copy()
    coords = sub_coords_df[["x", "y"]].values
    coords = scaler.transform(coords)
    # if fig_type is not None:
    #     save_fig_dir = "{0}/{1}_figures/".format(input_dir, fig_type) if fig_type is not None else "{0}/figures/".format(input_dir)
    # else:
    #     save_fig_dir = input_dir
    # makedirs(save_fig_dir)
    
    # Divide the images into bins 
    bins = np.linspace(0, 1, n_bins+1)
    ins_x = np.digitize(coords[:, 0], bins)
    ins_x[ins_x>n_bins] = n_bins
    ins_y = np.digitize(coords[:, 1], bins)
    ins_y[ins_y>n_bins] = n_bins

    grid_txt = ["{0}|{1}".format(ins_yi, ins_xi) for ins_xi, ins_yi in zip(ins_x, ins_y)] #transpose to adapt to the grid indices 
    grid_size= 1/n_bins
    n_col = n_bins + 2*int(margin/grid_size)
    n_row = n_bins + 2*int(margin/grid_size)

    # fig = plt.figure(figsize=(15, 15))
    fig = plt.figure(figsize=(int(3*n_bins), int(3*n_bins)), dpi=300)
    grid = plt.GridSpec(n_row, n_col)
    grid.update(wspace=0., hspace=0.)
    plt.rcParams["axes.linewidth"] = 1

    # Plot on the background 
    # sns.kdeplot(coords[:, 0], coords[:, 1], shade=True, cmap="Oranges")
    if bck_var is not None:
        all_coords = scaler.transform(coords_df[["x", "y"]].values) 
        x = all_coords[:, 0]
        y = all_coords[:, 1]
        z = coords_df[bck_var].values
        f = Rbf(x, y, z, kind="gaussian", smooth=5, episilon=5)
        x_grid = np.linspace(-margin, 1+margin, 200)
        y_grid = np.linspace(-margin, 1+margin, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y) 
        plt.pcolormesh(X, Y, Z, alpha=0.8, vmin=0, vmax=170, cmap=plt.cm.YlGn, shading="gouraud")
        plt.scatter(x, y, color="black", s=1000, edgecolor="white", linewidths=5)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0-margin, 1+margin])
    plt.ylim([0-margin, 1+margin])
    plt.box(False)
    
    # Show the image representing in each bin
    for gt in np.unique(grid_txt):
        check_idxes = [i for i in np.arange(len(grid_txt)) if grid_txt[i]==gt]
        sub_df = sub_coords_df.iloc[check_idxes]
        check_coords = coords[check_idxes]
        ref_coords = np.mean(check_coords, axis=0)
        check_img_idx = np.argmin(np.mean(abs(check_coords-ref_coords)))
        check_img_file = sub_df.iloc[check_img_idx]["file"]
        if "/" in check_img_file:
            img = plt.imread(check_img_file)
        else:
            img = plt.imread("{0}/{1}".format(input_dir, check_img_file))
       

        grid_coords = np.array(gt.split("|")).astype(np.int32) - 1 + int(margin/grid_size)
        # print(check_img_file, grid_coords)
        ax = fig.add_subplot(grid[n_row-grid_coords[0], grid_coords[1]])
        ax.margins(0, 0)
        ax.imshow(img, cmap="Greys") # show gray images
        if w_annot:
            txt = ".".join(e for e in check_img_file.split("/")[-1].split(".")[:-1])
            ax.annotate(txt, (0.5, 0.5), xycoords='axes fraction', fontsize=int(0.8*n_bins), ha="center", va="center")
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.autoscale_view('tight')
    if w_annot: 
        if more_info is not None:
            more_info += "_wAnnot"
        else:
            more_info = "wAnnot"

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if save_fig_dir is not None: 
        if more_info is not None:
            save_file = "{0}/img_map_nbins{1}_{2}.pdf".format(save_fig_dir, n_bins, more_info)
        else:
            save_file = "{0}/img_map_nbins{1}.pdf".format(save_fig_dir, n_bins)
        plt.savefig(save_file, transparent=False)
        print('Save at: {}'.format(save_file))
        release_mem(fig)
    else: 
        plt.show()  

def mlkr_dts(dat_df, xlim=[-0.1, 1.1], ylim=[-0.1, 1.1], bck_var=None, margin=0.1, more_info=None, save_dir=None):
    dat_df["dts_idx"] = [int(fi.split("_")[0]) for fi in dat_df.index]
    dts_fig = plt.figure(figsize=(8, 8))
    ax = dts_fig.add_subplot(111)

    if bck_var is not None:
        x = dat_df["x"].values
        y = dat_df["y"].values
        z = dat_df[bck_var].values
        f = Rbf(x, y, z, kind="gaussian", smooth=5, episilon=5)
        x_grid = np.linspace(-margin, 1+margin, 200)
        y_grid = np.linspace(-margin, 1+margin, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y) 
        plt.pcolormesh(X, Y, Z, alpha=0.8, vmin=0, vmax=170, cmap=plt.cm.YlGn, shading="gouraud")

    for dts_idx in sorted(set(dat_df["dts_idx"].values)):
        sub_df = dat_df[dat_df["dts_idx"]==dts_idx]
        ax.scatter(*sub_df[["x", "y"]].values.T, marker="o", c=dts_colors[dts_idx-1],
                    s=80, edgecolor="white", linewidths=0.4, label=str(dts_idx))
    plt.legend(loc="best", fontsize=18)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    dts_fig.subplots_adjust(left=0.0125, bottom=0.0125, right=.9875, top=.9875, wspace=0.0, hspace=0.0)
    if save_dir is not None:
        more_info = "" if more_info is None else "_"+more_info
        makedirs(save_dir)
        save_name = save_dir+"MLKR_map_dataset{}.pdf".format(more_info)
        plt.savefig(save_name, transparent=False)
        print("Save at: {}".format(save_name))
    else: 
        plt.show()
    release_mem(dts_fig)

if __name__=="__main__":    
    # gen_features_df = pd.read_csv("./output/GAN_gen_random/GAN_features.csv", index_col=0)
    # obs_features_df = pd.read_csv("./output/obs_NCs/GAN_features.csv", index_col=0)
    # obs_features_df["k"] = all_features_df.loc[obs_features_df.index, "k"].values
    # obs_features_df["G"] = all_features_df.loc[obs_features_df.index, "G"].values
    # mlkr_gen(obs_features_df, gen_features_df, margin=0.1, save_fig="./Visualize/mlkr_maps/mlkr_map_cropNC.pdf")
    # mlkr_gen(all_features_df, gen_features_df, save_fig="./Visualize/mlkr_maps/mlkr_map.pdf")

    mlkr_dts(all_features_df, bck_var="k", more_info="wBck", save_dir="../Visualize/mlkr_maps/")

    # coords_df = all_features_df[["x", "y", "k"]]
    # img_dir = "./input/obs_NCs/"
    # coords_df["file"] = ["{0}/{1}.png".format(img_dir, fi) for fi in coords_df.index]
    # images_scatter(coords_df, n_bins=25, w_annot=False, save_fig_dir="./Visualize/mlkr_maps/", more_info=None)

    # mlkr_scatter_pair(all_features_df, selected_pairs, bck_var="k", color_set=["red", "blue"], 
    #                 marker_set=["o", "o"], sct_lw=2.5, sct_size=50, sct_alpha=1, 
    #                 save_fig="./Visualize/mlkr_pair_wVector.pdf")
    
    # mlkr_cnt(all_features_df, w_annot=False,
    #         save_fig="./Visualize/mlkr_contour.pdf")