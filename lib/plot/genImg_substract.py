import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial.distance import euclidean, cityblock
from itertools import product

from refine_plot_mlkr import unit_dict, variable_signs
lib_dir = "./lib/"
if not lib_dir in sys.path:
    sys.path.append(lib_dir)
from general_lib import *

# ref_img = cv.imread("./Visualize/crop_NCs_org/1_517.png")
ref_img = cv.imread("./input/obs_NCs/1_517.png")
ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
_, ref_thresh = cv.threshold(ref_gray, 230, 255, cv.THRESH_BINARY)
ref_thresh[ref_thresh==0] = 1
ref_thresh[ref_thresh==255] = 0

all_features_df = pd.read_csv("./output/exp_data/All_datasets_features.csv", index_col=0)
gen_features_df = pd.read_csv("./output/GAN_gen_featureAxis/GAN_features.csv", index_col=0)
srt_cols = [f for f in all_features_df.columns if f in gen_features_df.columns]
gen_features_df = gen_features_df.reindex(columns=srt_cols)

def process_delta_img(hm, change_thresh=None, adjust_center=False, gaussian_sigma=1):
    if change_thresh is not None:
        tmp_heatmap = np.zeros(hm.shape)
        tmp_heatmap[hm<-change_thresh] = -1
        tmp_heatmap[hm>change_thresh] = 1
    else:
        tmp_heatmap = hm.copy()
    if adjust_center:
        fill_rows = np.where(np.sum(tmp_heatmap, axis=1)!=0)[0]
        fill_cols = np.where(np.sum(tmp_heatmap, axis=0)!=0)[0]
        crop_dims = [fill_rows.max()-fill_rows.min(), fill_cols.max()-fill_cols.min()]
        tmp_hm = np.zeros(tmp_heatmap.shape)
        tmp_hm[int(0.5*(tmp_hm.shape[0]-crop_dims[0])):int(0.5*(tmp_hm.shape[0]+crop_dims[0])), 
                int(0.5*(tmp_hm.shape[1]-crop_dims[1])):int(0.5*(tmp_hm.shape[1]+crop_dims[1]))] = \
                    tmp_heatmap[fill_rows.min():fill_rows.max(), fill_cols[0]:fill_cols.max()]
        tmp_heatmap = tmp_hm.copy()
    if gaussian_sigma>0:
        tmp_heatmap = gaussian_filter(tmp_heatmap, sigma=gaussian_sigma, mode="reflect", truncate=8)
    return tmp_heatmap 

def get_feature_txt(feature, org_df, mode="info", diff_method="diff", left_pad=0, right_pad=0):
    assert mode in ["info", "diff_info"]
    assert diff_method in ["dev", "diff"]
    if len(org_df)>1:
        if diff_method=="diff":
            diff_val = org_df[feature].values[-1] - org_df[feature].values[0]
            rel_diff_val = 100*abs(diff_val/(1e-6+org_df[feature].values[0])) 
        else:
            diff_val = org_df[feature].values.max() - org_df[feature].values.min()
            rel_diff_val = 100*abs(diff_val/(1e-6+org_df[feature].values.min())) 
    else:
        diff_val = 0
        rel_diff_val = 0
    f_sign = "+" if diff_val>=0 else "-"
    txt_color = "red" if f_sign=="+" else "blue"
    # if mode=="info":
    #     txt = left_pad*" " + r"{}".format(variable_signs[feature]) + \
    #         "={0:.2f}".format(org_df[feature].values[-1]) + unit_dict[feature] + \
    #         " ({0}{1:.1f}, {0}{2:.1f}%)".format(f_sign, abs(diff_df[feature]), rel_diff_val) + right_pad*" "
    # else:
        # rel_diff_val = 100*abs(diff_df[feature]/(1e-6+org_df[feature].values[0])) 
    diff_val = np.max([abs(diff_val), 0.1]) if round(rel_diff_val, 1)>0 else 0
    txt = left_pad*" " + r"$\Delta$" +r"{}".format(variable_signs[feature]) + \
        "={0:.1f}".format(diff_val) + unit_dict[feature] + \
        " ({0}{1:.1f}%)".format(f_sign, rel_diff_val) + right_pad*" "
    return txt, txt_color

def draw_canvas(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))
    X[:, :, [0, 2]] = X[:, :, [2, 0]]
    return X

def preprocess_genImg(gen_img_dir, ref_img=ref_img):
    # imgs = []
    save_img_dir = gen_img_dir+"/adjust_imgs/"
    makedirs(save_img_dir)
    check_imgs = [fn for fn in os.listdir(gen_img_dir) if ".png" in fn]
    for i, img_name in enumerate(check_imgs):
        img = cv.imread("{0}/{1}".format(gen_img_dir, img_name))
        img = cv.resize(img, ref_img.shape[:2])
        crop_img = img.copy()
        gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
        thresh[thresh==0] = 1
        thresh[thresh==255] = 0
        crop_height = np.where(np.sum(thresh, axis=1)!=0)[0][[0, -1]]
        crop_width = np.where(np.sum(thresh, axis=0)!=0)[0][[0, -1]]
        crop_img = crop_img[crop_height[0]:crop_height[1]+1, crop_width[0]:crop_width[1]+1]
        crop_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        _, crop_thresh = cv.threshold(crop_gray, 230, 255, cv.THRESH_BINARY)

        refine_img = np.zeros(img.shape).astype(np.int32)
        shift_y = int(0.5*(img.shape[0]-crop_img.shape[0]))
        # if i==0: 
        #     shift_x = int(0.5*(img.shape[1]-crop_img.shape[1]))
        #     x_ref = img.shape[1]-crop_img.shape[1]-shift_x
        # else:
        x_ref = ref_img.shape[1]-np.where(np.sum(ref_thresh, axis=0)!=0)[0][-1]-1
        shift_x = img.shape[1]-crop_img.shape[1]-x_ref
        refine_img[shift_y:crop_img.shape[0]+shift_y, shift_x:crop_img.shape[1]+shift_x, :] += 255-crop_img.astype(np.int32)
        refine_img = 255-refine_img
        cv.imwrite(save_img_dir+img_name, refine_img)

def save_rawImgs(img_dir, fd=6, fdpi=200):
    check_imgs = [fn for fn in os.listdir(img_dir) if ".png" in fn]
    for i, img_name in enumerate(check_imgs):
        img = cv.imread("{0}/{1}".format(img_dir, img_name))
        fig = plt.figure(figsize=(fd, fd), dpi=fdpi)
        ax = fig.add_subplot(111)
        zoom_img = cv.resize(img, (fd*fdpi, fd *fdpi))
        ax.imshow(zoom_img)
        plt.savefig(img_dir+img_name.replace(".png", ".pdf"))

def plot_genDiff_info(img, sub_df=None, img_cat=True, video_magnification=0.0384, 
                    diff_method="diff", color_val=20, fig_dim=6, fig_dpi=200, 
                    cmap="bwr", img_type="Observed", fs=18):
    assert img.shape[0]==img.shape[1]
    zoom_img = cv.resize(img, (fig_dim*fig_dpi, fig_dim*fig_dpi))
    crop_size = zoom_img.shape[0]
    offset = int(crop_size/40)
    fig = plt.figure(figsize=(fig_dim, fig_dim), dpi=fig_dpi)
    ax = fig.add_subplot(111)
    c_val = 1 if img_cat else color_val
    # c_val = np.max([ref_cval, np.max(abs(img))])
    # c_val = ref_cval
    if c_val is not None:
        ax.imshow(zoom_img, vmin=-c_val, vmax=c_val, cmap=cmap)
    else:
        ax.imshow(zoom_img, cmap=cmap)

    if sub_df is not None:
        ax.annotate(img_type, (crop_size-offset, offset), fontsize=fs, ha="right", va="top")
        ax.plot([crop_size-offset-(1*(zoom_img.shape[0])/(img.shape[0]*video_magnification)), crop_size-offset], [5.5*offset, 5.5*offset], color="gray", lw=2.5)
        ax.annotate("1 nm", (crop_size-offset, 3.5*offset), fontsize=fs, ha="right", va="top")
        
        ax.annotate("Integral", (offset, offset), fontsize=fs, va="top") 
        ax.annotate("{0}".format(sub_df.index[-1]), (offset, 3.5*offset), fontsize=16, va="top")
        # ax.annotate("{0}s".format(round(float(sub_df.index[0].split("_")[1])*1/30, 2)), (offset, 3.5*offset), fontsize=16, va="top")

        axins = ax.inset_axes([crop_size-3*offset, 7.5*offset, offset, 10*offset], transform=ax.transData, zorder=10)
        norm = mpl.colors.Normalize(vmin=-c_val, vmax=c_val)
        cb = mpl.colorbar.ColorbarBase(axins, cmap=eval("mpl.cm.{}".format(cmap)), norm=norm, orientation='vertical')
        axins.tick_params(axis='y', which='major', direction="out", labelsize=12, left=True, right=False, labelleft=True, labelright=False)

        txt, txt_color =  get_feature_txt("k", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (crop_size*0.3, offset), color=txt_color, fontsize=fs, ha="left", va="top")
        txt, txt_color =  get_feature_txt("G", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (crop_size*0.3, 3.5*offset), color=txt_color, fontsize=fs, ha="left", va="top")

        txt, txt_color =  get_feature_txt("d_min", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (offset, crop_size-7*offset), color=txt_color, fontsize=fs, va="bottom")
        txt, txt_color =  get_feature_txt("X_base_l", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (offset, crop_size-4*offset), color=txt_color, fontsize=fs, va="bottom")
        txt, txt_color =  get_feature_txt("Y_base_l", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt,  (offset, crop_size-offset), color=txt_color, fontsize=fs, va="bottom")

        txt, txt_color =  get_feature_txt("mean_D", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (crop_size-offset, crop_size-7*offset), color=txt_color, fontsize=fs, ha="right", va="bottom")
        txt, txt_color =  get_feature_txt("var_D", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (crop_size-offset, crop_size-4*offset), color=txt_color, fontsize=fs, ha="right", va="bottom")
        txt, txt_color =  get_feature_txt("F_D", sub_df, mode="diff_info", diff_method=diff_method)
        ax.annotate(txt, (crop_size-offset, crop_size-offset), color=txt_color, fontsize=fs, ha="right", va="bottom")
    
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    return fig 

def get_image_map(gen_img_dir, diff_x=0.4, diff_y=0.2, err_thresh=0.1, n_bins=5, ref_fr="1_517"):
    features_df = pd.read_csv(gen_img_dir+"GAN_features.csv", index_col=0)
    n_bins = [n_bins, n_bins] if type(n_bins)==int else n_bins
    diff_x_range = np.round_(np.linspace(-diff_x, diff_x, n_bins[0]).astype(float), 2)
    diff_y_range = np.round_(np.linspace(-diff_y, diff_y, n_bins[1]).astype(float), 2)
    ref_xy = all_features_df.loc[ref_fr, ["X_base_l", "Y_base_l"]].values

    diff_features_df = features_df[["X_base_l", "Y_base_l"]].copy()
    diff_features_df["X_base_l"] = diff_features_df["X_base_l"].values-ref_xy[0]
    diff_features_df["Y_base_l"] = diff_features_df["Y_base_l"].values-ref_xy[1]
    err_thresh_x = np.min([err_thresh, diff_x/(n_bins[0]-1)])
    err_thresh_y = np.min([err_thresh, diff_y/(n_bins[1]-1)])
    diff_labels = []
    for xi, yi in diff_features_df[["X_base_l", "Y_base_l"]].values:
        diff_xi = np.min(abs(diff_x_range-xi))
        diff_yi = np.min(abs(diff_y_range-yi))
        if diff_xi<err_thresh_x and diff_yi<err_thresh_y:
            diff_txt = "{0:.1}|{1:.1}".format(diff_x_range[np.argmin(abs(diff_x_range-xi))], \
                                            diff_y_range[np.argmin(abs(diff_y_range-yi))])
            diff_txt = diff_txt.replace("0e+00", "0.0") if "0e+00" in diff_txt else diff_txt
        else:
            diff_txt = "None"
        diff_labels += [diff_txt]
    diff_features_df["label"] = diff_labels

    grid_idxes = list(product(np.arange(n_bins[0]), np.arange(n_bins[1])))
    map_df = pd.DataFrame(index=diff_y_range, columns=diff_x_range)
    map_df.loc[0.0, 0.0] = "{0} {1} {2}".format(ref_fr, 0, 0)
    for grid_idx in grid_idxes:
        grid_x = grid_idx[0]
        grid_y = grid_idx[1]
        if grid_x!=int(n_bins[0]/2) or grid_y!=int(n_bins[1]/2):
            grid_txt = "{0:.1}|{1:.1}".format(diff_x_range[grid_x], diff_y_range[grid_y])
            grid_txt = grid_txt.replace("0e+00", "0.0") if "0e+00" in grid_txt else grid_txt
            sub_df = diff_features_df[diff_features_df["label"]==grid_txt]
            diff_xyi = np.array([diff_x_range[grid_x], diff_y_range[grid_y]])

            if len(sub_df)>0:
                sub_df["diff_err"] = [cityblock(xyi, diff_xyi) for xyi in sub_df[["X_base_l", "Y_base_l"]].values]
                sub_df = sub_df.sort_values(by="diff_err", ascending=True)
                select_img = sub_df.index[0]
                map_df.loc[diff_y_range[grid_y], diff_x_range[grid_x]] = \
                    "{0} {1} {2}".format(select_img, *sub_df.loc[select_img, ["X_base_l", "Y_base_l"]].values)
            else:
                map_df.loc[diff_y_range[grid_y], diff_x_range[grid_x]] = None
    return map_df

def plot_image_map(gen_img_dir, map_df=None, diff_x=0.4, diff_y=0.2, n_bins=5, err_thresh=0.1, ref_fr="1_517", 
                gaussian_sigma=1, cval=20, fd=6, fdpi=200, w_text=True, save_subfig=True, save_fig_dir=None):
    img_dir = gen_img_dir+"/adjust_imgs/"
    if not os.path.isdir(img_dir): 
        preprocess_genImg(gen_img_dir)
    if map_df is None:
        n_bins = [n_bins, n_bins] if type(n_bins)==int else n_bins
        map_df = get_image_map(gen_img_dir, diff_x=diff_x, diff_y=diff_y, err_thresh=err_thresh, n_bins=n_bins, ref_fr=ref_fr)
    else:
        n_bins = [len(map_df.columns), len(map_df.index)]
    features_df = pd.concat([gen_features_df.loc[sorted(gen_features_df.index)], all_features_df.loc[[ref_fr], srt_cols]])
    ref_img = cv.imread("./input/obs_NCs/{}.png".format(ref_fr))
    ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    _, ref_thresh = cv.threshold(ref_gray, 230, 255, cv.THRESH_BINARY)
    ref_thresh = (255-ref_thresh.astype(float))/255
    zoom_ref_img = cv.resize(ref_img, (fd*fdpi, fd*fdpi))
    _, _, _, _, ref_cnt_surface = process_frame(zoom_ref_img, gray_thresh=230, smooth_deg=1, noise_tol=7, draw_contour=1)
    ref_cnt_surface = np.squeeze(ref_cnt_surface)

    # get images
    img_names = [img_info.split(" ")[0] for img_info in map_df.values.ravel() if img_info is not None]
    if save_subfig:
        save_img_dir = save_fig_dir+"/subtract_imgs_wText/" if w_text else save_fig_dir+"/subtract_imgs/"
        makedirs(save_img_dir)
    img_dict = {}
    for img_name in img_names:
        print(img_name)
        if img_name!=ref_fr:
            img = cv.imread("{0}/{1}.png".format(img_dir, img_name))

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
            thresh = (255-thresh.astype(float))/255
            hm_filter = thresh.astype(float)+ref_thresh.astype(float)
            trim_x = np.where(np.max(hm_filter, axis=0)==2)[0].min()
            hm_filter[:, :trim_x] = np.zeros((hm_filter.shape[0], trim_x))

            hm = ref_gray.astype(np.float)-gray.astype(np.float)
            hm[hm_filter==0] = 0
            hm[(hm_filter==1)&(hm<=0)] = 0
            heatmap = process_delta_img(hm, gaussian_sigma=gaussian_sigma)
            
            check_df = features_df.loc[[ref_fr, img_name.replace(".png", "")]]
            fig = plot_genDiff_info(heatmap, sub_df=check_df if w_text else None, img_cat=False, 
                                    color_val=cval, fig_dim=fd, fig_dpi=fdpi, fs=16, cmap="bwr", img_type="GAN-gen")
        else:
            fig = plt.figure(figsize=(fd, fd), dpi=fdpi)
            ax = fig.add_subplot(111)
            ax.imshow(zoom_ref_img, cmap="gray")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.plot(*ref_cnt_surface.T, lw=6, color="chartreuse", zorder=10, alpha=1.0)
        if save_subfig:
            plt.savefig("{0}/{1}.pdf".format(save_img_dir, img_name), transparent=True)
        X = draw_canvas(fig)
        X[:, :, [0, 2]] = X[:, :, [2, 0]]
        img_dict[img_name] = X[:, :, :3]

    fig = plt.figure(figsize=(int(3*n_bins[0]), int(3*n_bins[1])), dpi=300)
    grid = plt.GridSpec(n_bins[1], n_bins[0])
    grid.update(wspace=0.5, hspace=0.5)
    plt.rcParams["axes.linewidth"] = 1
    grid_idxes = list(product(np.arange(n_bins[0]), np.arange(n_bins[1])))
    for grid_idx in grid_idxes:
        grid_x = grid_idx[0]
        grid_y = grid_idx[1] 
        img_info = map_df.values[grid_y, grid_x]
        if img_info is not None:
            ax = fig.add_subplot(grid[grid_y, grid_x])
            img = img_dict[img_info.split(" ")[0]]
            ax.imshow(img)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.autoscale_view('tight')
    grid.tight_layout(fig)
    if save_fig_dir is not None:
        makedirs(save_fig_dir)
        more_info = "_wText" if w_text else ""
        if map_df is None:
            save_name = "diffX{0}_diffY{1}_nBins{2}_errThresh{3}_Image{4}.pdf".format(diff_x, diff_y, n_bins, err_thresh, more_info)
        else:
            save_name = "diffX{0}_diffY{1}_nBins{2}_Image{3}.pdf".format(map_df.columns[-1], map_df.index[-1], n_bins, more_info)
        plt.savefig(os.path.join(save_fig_dir, save_name))
    else:
        plt.show()

def plot_map_info(gen_img_dir, map_df=None, check_feature="k", diff_x=0.4, diff_y=0.2, n_bins=5, err_thresh=0.1, ref_fr="1_517", 
                cval=None, cmap="bwr", w_info=False, save_fig_dir=None):
    var_names = {"d_min":"r_min", "X_base_l":"delta_x", "Y_base_l":"delta_y"}
    if not check_feature in var_names:
        var_names[check_feature] = check_feature
    if map_df is None:
        n_bins = [n_bins, n_bins] if type(n_bins)==int else n_bins
        map_df = get_image_map(gen_img_dir, diff_x=diff_x, diff_y=diff_y, err_thresh=err_thresh, n_bins=n_bins, ref_fr=ref_fr)
    else:
        n_bins = [len(map_df.columns), len(map_df.index)]
    # map_df = get_image_map(gen_img_dir, diff_x=diff_x, diff_y=diff_y, err_thresh=err_thresh, n_bins=n_bins, ref_fr=ref_fr)
    selected_imgs = [img_info.split(" ")[0] for img_info in map_df.values.ravel() if img_info is not None]
    selected_imgs = np.unique([img_name for img_name in selected_imgs if img_name!=ref_fr])
    features_df = pd.read_csv(gen_img_dir+"GAN_features.csv", index_col=0)
    srt_cols = [f for f in all_features_df.columns if f in features_df.columns]
    diff_features_df = features_df.loc[selected_imgs, srt_cols] - all_features_df.loc[ref_fr, srt_cols]
    cval = np.max(abs(diff_features_df[check_feature].values)) if cval is None else cval

    fig = plt.figure(figsize=(int(3*n_bins[0]), int(3*n_bins[1])), dpi=300)
    grid = plt.GridSpec(n_bins[1], n_bins[0])
    grid.update(wspace=0.05, hspace=0.05)
    plt.rcParams["axes.linewidth"] = 1
    grid_idxes = list(product(np.arange(n_bins[0]), np.arange(n_bins[1])))
    for grid_idx in grid_idxes:
        grid_x = grid_idx[0]
        grid_y = grid_idx[1]
        img_info = map_df.values[grid_y, grid_x]
        if img_info is not None:
            ax = fig.add_subplot(grid[grid_y, grid_x])
            img_name = img_info.split(" ")[0]
            if img_name in diff_features_df.index:
                diff_val = diff_features_df.loc[img_name, check_feature]
                img = diff_val*np.ones((900, 900))
            else:
                diff_val = all_features_df.loc[ref_fr, check_feature]
                img = np.zeros((900, 900))
            ax.imshow(img, cmap=cmap, vmin=-cval, vmax=cval)
            if w_info:
                val_sign = "+" if diff_val>=0 and img_name!=ref_fr else ""
                ax.annotate(val_sign+str(np.round_(diff_val, 1)), (0.5, 0.5), xycoords='axes fraction', fontsize=60, ha="center", va="center")
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            # ax.autoscale_view('tight')
    # if w_info:
    #     cax = fig.add_subplot(grid[:, -1])
    #     norm = mpl.colors.Normalize(vmin=-cval, vmax=cval)
    #     cb = mpl.colorbar.ColorbarBase(cax, cmap=eval("mpl.cm.{}".format(cmap)), norm=norm, orientation='vertical')
    #     cax.tick_params(axis='y', which='major', direction="out", labelsize=30, left=False, right=True, labelleft=False, labelright=True)
    # grid.tight_layout(fig)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    if save_fig_dir is not None:
        makedirs(save_fig_dir)
        more_info = "_wInfo" if w_info else ""
        save_name = "{0}_diffX{1}_diffY{2}_nBins{3}_errThresh{4}{5}.pdf".format(var_names[check_feature], diff_x, diff_y, n_bins, \
                                                                                err_thresh, more_info)
        plt.savefig(os.path.join(save_fig_dir, save_name))
    else:
        plt.show()

if __name__=="__main__":
    img_map = np.array([["-0.75|0.4", "-0.25|1.25", "-0.0|0.55", "0.1|0.9", "0.25|2.05"], 
                        ["-0.7|0.3", "-0.2|0.65", "1_517", "0.05|-0.05", "0.15|0.05"], 
                        ["-0.65|0.1", "-0.3|0.8", "0.05|0.4", "0.05|0.45", "0.05|-0.15"]])
    img_map_df = pd.DataFrame(img_map, columns=np.round_(np.linspace(-0.4, 0.4, 5).astype(float), 2),
                            index=np.round_(np.linspace(-0.1, 0.1, 3).astype(float), 2))

    print(img_map_df)
    plot_image_map(gen_img_dir="./output/GAN_gen_featureAxis/refine_imgs/", map_df=img_map_df, w_text=False, cval=20, 
                    save_subfig=True, save_fig_dir="./Visualize/diff_map/")
    for fi in ["k", "G", "d_min", "X_base_l", "Y_base_l", "mean_D", "var_D", "F_D"]:
        plot_map_info(gen_img_dir="./output/GAN_gen_featureAxis/", map_df=img_map_df, check_feature=fi, w_info=True,  
                    cval=50 if fi=="k" else None, 
                    # save_fig_dir=None)
                    save_fig_dir="./Visualize/diff_map/diff_feature_map/")