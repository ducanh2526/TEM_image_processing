import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

lib_dir = "./lib/"
if not lib_dir in sys.path:
    sys.path.append(lib_dir)
from general_lib import *
from VarGen import video_names
from refine_plot_mlkr import variable_signs, images_scatter

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def gen_img_dict(save_dir="./Visualize/"):
    bck_img_dict = {}
    for i, video_name in enumerate(video_names):
        cap_dict = get_cap_info("./input/exp_data/{0}/{0}_background.mp4".format(video_name))
        cap = cap_dict["cap"]
        base_pos_df = pd.read_csv("./output/exp_data/{0}/{0}_base_pos.csv".format(video_name), index_col=0)
        for fr_idx in range(cap_dict["totalFrames"]):
            base_pos = base_pos_df.loc["f{}".format(fr_idx)].values
            cap.set(cv.CAP_PROP_POS_FRAMES, fr_idx)
            _, fr = cap.read()
            fr_gray = cv.cvtColor(fr.astype(np.uint8), cv.COLOR_BGR2GRAY)
            _, fr_thresh = cv.threshold(fr_gray, 230, 255, cv.THRESH_BINARY)
            fr[fr_thresh==255] = (255, 255, 255)
            bck_img_dict["{0}_{1}".format(i+1, fr_idx)] = {"img":fr, "base_pos":base_pos}
    with open(save_dir+"/All_datasets_imgs.pickle", "wb") as handle:
        pickle.dump(bck_img_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
img_dict_dir = "./Visualize/All_datasets_imgs.pickle"
if not os.path.isfile(img_dict_dir):
    gen_img_dict(save_dir="./Visualize")
with open(img_dict_dir, "rb") as handle:
    img_dict = pickle.load(handle)

def pairing_frs(features_df, x_thresh=0.2, y_filter=None, fr_thresh=None, labels=["push", "pull"], 
                save_dir="./TEM_GAN/GAN_interpolate/"):
    makedirs(save_dir)
    save_name = save_dir+"fr_pair_xThresh{0}_yFilter{1}_frThresh{2}.csv".format(x_thresh, y_filter, fr_thresh)
    if not os.path.isfile(save_name):
        dts_idxes = np.array([int(fn.split("_")[0]) for fn in features_df.index])
        if y_filter is not None: 
            labels = ["straight_"+li for li in labels]
        else: 
            pass
        y_filter = y_filter if y_filter is not None else 9999
        selected_pairs_info = []
        for dts_idx in sorted(set(np.array(dts_idxes))):
            check_idx = np.where(dts_idxes==dts_idx)[0]
            fr_names = np.array([int(fn.split("_")[1]) for fn in features_df.index[check_idx]])
            srt_check_idx = check_idx[np.argsort(fr_names)]
            sub_df = features_df.iloc[srt_check_idx]
            for i in np.arange(len(sub_df)-1):
                curr_fr = sub_df.index[i]
                valid_df = sub_df.copy()
                valid_df = valid_df.iloc[i+1:i+fr_thresh+1] if fr_thresh is not None else valid_df.iloc[i+1:]
                valid_df["X_base_l"] = valid_df["X_base_l"].values-sub_df.loc[curr_fr, "X_base_l"]
                valid_df["Y_base_l"] = valid_df["Y_base_l"].values-sub_df.loc[curr_fr, "Y_base_l"]
                valid_df = valid_df.loc[(abs(valid_df["X_base_l"])>=x_thresh)]
                if len(valid_df)>0:
                    valid_fr_idxes = np.array([int(fn.split("_")[1]) for fn in valid_df.index])
                    nxt_fr = valid_df.index[np.argmin(valid_fr_idxes)]
                    dev_x = valid_df.loc[nxt_fr, "X_base_l"]
                    dev_y = valid_df.loc[nxt_fr, "Y_base_l"]
                    if abs(dev_y)<=y_filter:
                        selected_pairs_info += [{"curr_fr":curr_fr, "nxt_fr":nxt_fr, 
                                                "delta_x":dev_x, "delta_y":dev_y, 
                                                "label":labels[0] if dev_x<0 else labels[1]}]
                else:
                    pass
        pair_df = pd.DataFrame.from_dict(selected_pairs_info, orient="columns")
        pair_df.to_csv(save_name)
    pair_df = pd.read_csv(save_name, index_col=0)
    return pair_df

def delta_img(curr_fr, nxt_fr):
    curr_gray = cv.cvtColor(curr_fr, cv.COLOR_BGR2GRAY)
    nxt_gray = cv.cvtColor(nxt_fr, cv.COLOR_BGR2GRAY)
    heatmap = -(nxt_gray.astype(np.float)-curr_gray.astype(np.float)) # inverse since higher gray scale ~ less density
    return heatmap

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

def get_dim_crop(dts_idx, crop_size=250):
    ref_fr_name = "{}_0".format(dts_idx)
    ref_img = img_dict[ref_fr_name]["img"]
    base_pos = img_dict[ref_fr_name]["base_pos"]
    tmp_img = np.zeros(ref_img.shape)
    tmp_img[:, base_pos[0]: base_pos[1]] += 255-ref_img[:, base_pos[0]: base_pos[1]]
    fr = 255-tmp_img
    fr_gray = cv.cvtColor(fr.astype(np.uint8), cv.COLOR_BGR2GRAY)
    _, fr_thresh = cv.threshold(fr_gray, 230, 255, cv.THRESH_BINARY)
    fill_rows = np.where(np.sum(255-fr_thresh[:, base_pos[0]:base_pos[1]], axis=1)!=0)[0]
    height_lim = np.array(fill_rows[[0, -1]])
    width_crop = [int(np.mean(base_pos)-crop_size/2), int(np.mean(base_pos)+crop_size/2)]
    height_crop =  [int(np.mean(height_lim)-crop_size/2), int(np.mean(height_lim)+crop_size/2)]
    return [height_crop, width_crop]

def gen_single_heatmap(fr_queue):
    dts_idx = int(fr_queue[0].split("_")[0])
    check_imgs = [img_dict[fn]["img"] for fn in fr_queue]
    base_pos = [img_dict[fn]["base_pos"] for fn in fr_queue]
    dim_crop = get_dim_crop(dts_idx=dts_idx, crop_size=250)

    curr_img = check_imgs[0]
    curr_gray = cv.cvtColor(curr_img.astype(np.uint8), cv.COLOR_BGR2GRAY)
    _, curr_thresh = cv.threshold(curr_gray, 230, 255, cv.THRESH_BINARY)
    nxt_img = check_imgs[1]
    nxt_gray = cv.cvtColor(nxt_img.astype(np.uint8), cv.COLOR_BGR2GRAY)
    _, nxt_thresh = cv.threshold(nxt_gray, 230, 255, cv.THRESH_BINARY)
    tmp_img = np.zeros(curr_img.shape[:2])
    tmp_img[:, base_pos[0][0]: base_pos[0][1]] += 255-curr_thresh[:, base_pos[0][0]: base_pos[0][1]]
    tmp_img[:, base_pos[1][0]: base_pos[1][1]] += 255-nxt_thresh[:, base_pos[1][0]: base_pos[1][1]]

    delta_heatmap = delta_img(curr_img, nxt_img)
    crop_heatmap = delta_heatmap[dim_crop[0][0]:dim_crop[0][1], dim_crop[1][0]:dim_crop[1][1]]
    crop_thresh = tmp_img[dim_crop[0][0]:dim_crop[0][1], dim_crop[1][0]:dim_crop[1][1]]/255
    
    return crop_heatmap, crop_thresh

def gen_subtract_imgs(query_pairs, gaussian_sigma=1, color_val=20, change_thresh=None, fd=6, fdpi=200, 
                    w_text=False, save_format="pdf", save_dir="./TEM_GAN/GAN_interpolate/Subtract_obs/"):
    save_format = "pdf" if w_text else save_format
    img_cat = not(change_thresh is None)
    makedirs(save_dir)
    for fr_pair in query_pairs:
        save_name = save_dir+"{0}|{1}.{2}".format(fr_pair[0], fr_pair[1], save_format)
        crop_heatmap, crop_thresh = gen_single_heatmap(fr_pair)
        crop_heatmap[crop_thresh==0] = 0
        crop_heatmap = process_delta_img(crop_heatmap, change_thresh=change_thresh, 
                                        adjust_center=True if np.sum(abs(crop_heatmap))>0 else False, gaussian_sigma=gaussian_sigma)
        
        fig = plt.figure(figsize=(fd, fd), dpi=fdpi)
        ax = fig.add_subplot(111)
        zoom_hm = cv.resize(crop_heatmap, (fd*fdpi, fd *fdpi))
        ax.imshow(zoom_hm, cmap="bwr", vmin=-color_val, vmax=color_val)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(save_name, dpi=300)
        release_mem(fig)
    
if __name__=="__main__":    
    all_features_df = pd.read_csv("./output/exp_data/All_datasets_features.csv", index_col=0)

    pair_frs = pairing_frs(all_features_df, x_thresh=0.2, y_filter=0.1, fr_thresh=50, save_dir="./TEM_GAN/GAN_interpolate/")
    pair_frs["dts_idx"] = [int(fn.split("_")[0]) for fn in pair_frs["curr_fr"].values]

    # for dts_idx in sorted(set(pair_frs["dts_idx"].values)):
    #     print(dts_idx)
    #     check_idx = np.where(pair_frs["dts_idx"].values==dts_idx)[0]
    #     sub_pair = pair_frs[pair_frs["dts_idx"]==dts_idx]
    #     gen_subtract_imgs(sub_pair[["curr_fr", "nxt_fr"]].values, color_val=20, 
    #                     save_format="png", save_dir="./Visualize/int_imgs/")
    
    state_dict = {"pull":{"color_set":["black", "orange"], "marker_set":[".", "s"]}, "push":{"color_set":["black", "cyan"], "marker_set":[".", "^"]}}
    for state in list(state_dict.keys()):
        if not state in pair_frs["label"].values:
            state_name = "straight_"+state
        selected_idx = np.where(pair_frs["label"].values==state_name)[0]
        selected_frs = pair_frs.iloc[selected_idx]["curr_fr"].values

        coords_df = pd.DataFrame(index=pair_frs["curr_fr"].values)
        for f in ["x", "y", "k"]:
            coords_df[f] = all_features_df.loc[pair_frs["curr_fr"].values, f].values
        coords_df["label"] = [1 if fn in selected_frs else 0 for fn in coords_df.index]
        img_names = []
        for i in np.arange(len(pair_frs)):
            fr_names = pair_frs[["curr_fr", "nxt_fr"]].values[i]
            dts_idx = int(fr_names[0].split("_")[0])
            img_name = "./Visualize/int_imgs/{1}|{2}.png".format(dts_idx, fr_names[0], fr_names[1])
            img_names += [img_name]
        coords_df["file"] = img_names
        for nbins in [20]:
            images_scatter(coords_df, "./Visualize/int_imgs/".format(state), 
                            check_inst=selected_frs, n_bins=nbins, 
                            bck_var="k", w_annot=False, more_info=state, 
                            save_fig_dir="./Visualize/mlkr_maps/")