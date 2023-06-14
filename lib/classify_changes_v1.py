import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from general_lib import *
from BaseDetect import video_names

all_dat_df = pd.DataFrame()
for dts_idx, video_name in enumerate(video_names):
    dat_df = pd.read_csv("./input/exp_data/{0}/{0}.csv".format(video_name), header=None)
    dat_df = dat_df.rename({0: 'fr', 1: 'G', 2:"k"}, axis='columns')
    dat_df["fr"] = dat_df["fr"].values-1
    dat_df = dat_df.set_index("fr")
    dat_df["dts_idx"] = [dts_idx+1 for rep in range(len(dat_df))]
    all_dat_df = pd.concat((all_dat_df, dat_df))

all_features_df = pd.read_csv("./output/exp_data/All_datasets_features.csv", index_col="fr_names")
print(all_features_df.head(5))

def gen_diff_record(dat_type="raw", window_size=6, step_size=6, filter_thresh=0, method="diff"):
    assert dat_type in ["raw", "filter"]
    assert method in ["diff", "dev"]

    all_features_df["dts_idx"] = [int(fn.split("_")[0]) for fn in all_features_df.index]

    val_per_fr = 80 if dat_type=="raw" else 1
    diff_features_df = pd.DataFrame()
    for dts_idx in sorted(set(all_dat_df["dts_idx"].values)):
        features_df = all_features_df[all_features_df["dts_idx"]==dts_idx]
        dat_df = all_dat_df[all_dat_df["dts_idx"]==dts_idx]
        check_df = dat_df.copy() if dat_type=="raw" else features_df.copy()
        if dat_type=="raw": 
            fr_names = ["{0}_{1}".format(dts_idx, int(fi)) for fi in \
                        check_df.index[np.arange(0, len(check_df)-window_size*val_per_fr, val_per_fr)]]
        else:
            fr_names = check_df.index[np.arange(0, len(check_df)-window_size*val_per_fr, val_per_fr)]
        diff_df = pd.DataFrame(index=fr_names)
        for feature in ["k", "G"]:
            if method=="dev":
                diff_df[feature] = [check_df.iloc[i:i+window_size*val_per_fr][feature].values.max()- \
                                    check_df.iloc[i:i+window_size*val_per_fr][feature].values.min() \
                                    for i in np.arange(0, len(check_df)-window_size*val_per_fr, val_per_fr)]
            else:
                diff_df[feature] = [check_df.iloc[i+window_size*val_per_fr][feature]- \
                                    check_df.iloc[i][feature] \
                                    for i in np.arange(0, len(check_df)-window_size*val_per_fr, val_per_fr)]

        check_frs = [fi for fi in diff_df.index if fi in features_df.index[np.arange(0, len(features_df)-window_size, step_size)]]
        diff_df = diff_df.loc[check_frs]
        check_fr_idxes = [int(fi.split("_")[1]) for fi in diff_df.index]
        check_frs = [fi for fi in diff_df.index if fi in features_df.index[np.arange(0, len(features_df)-window_size, step_size)]]
        diff_df = diff_df.loc[check_frs]
        check_fr_idxes = [int(fi.split("_")[1]) for fi in diff_df.index]
        if dat_type=="filter":
            # for feature in all_features_df.columns[:13]:
            for feature in ["k", "G"]:
                diff_df[feature] = [features_df.iloc[i:i+window_size][feature].values.max()- \
                                    features_df.iloc[i:i+window_size][feature].values.min() \
                                    for i in check_fr_idxes]
        diff_df["dts_idx"] = [dts_idx for rep in range(len(diff_df))]    
        diff_features_df = pd.concat([diff_features_df, diff_df])
    if filter_thresh>0:
        filter_df = all_features_df.loc[(all_features_df["G"]<=filter_thresh)]
        check_frs = []
        for fri in filter_df.index:
            dts_idx = int(fri.split("_")[0])
            fr_idx = int(fri.split("_")[1])
            if set(["{0}_{1}".format(dts_idx, fi) for fi in np.arange(fr_idx, fr_idx+window_size)]).issubset(set(filter_df.index)):
                check_frs += [fri]
        valid_frs = [fi for fi in diff_features_df.index if fi in check_frs]
        diff_features_df = diff_features_df.loc[valid_frs]
    return diff_features_df

def get_label(diff_df, Gk_thresh):
    G_thresh = Gk_thresh["G"]
    k_thresh = Gk_thresh["k"]
    labels = np.zeros((len(diff_df)))
    G_change_idx = np.where((abs(diff_df['G'])>G_thresh) & (abs(diff_df['k'])<=k_thresh))[0]
    labels[G_change_idx] = 1
    Gk_change_idx = np.where((abs(diff_df['G'])>G_thresh) & (abs(diff_df['k'])>k_thresh))[0]
    labels[Gk_change_idx] = 2
    k_change_idx = np.where((abs(diff_df['G'])<=G_thresh) & (abs(diff_df['k'])>k_thresh))[0]
    labels[k_change_idx] = 3
    return labels


if __name__=="__main__":
    diff_dat = "filter"
    window_size = 6
    step_size = 6
    Gfilter_thresh = 0
    fig_suffix =  "{0}Diff_ws{1}_ss{2}_filter{3}".format(diff_dat, window_size, step_size, Gfilter_thresh)
    thresh_df = pd.DataFrame(np.array([[20, 10], [15, 4], [15, 4], [10, 4], [15, 2]]), \
                            index=["D{}".format(di) for di in np.arange(1, 6)], \
                            columns=["k", "G"])
    cat_dict = {1:"G", 2:"G|k", 3:"k"}

    diff_record = gen_diff_record(dat_type=diff_dat, window_size=window_size, step_size=step_size, filter_thresh=Gfilter_thresh)

    label_record = pd.DataFrame()
    for dts_idx in np.arange(1, 6):
        prop_thresh = thresh_df.loc["D{}".format(dts_idx)].to_dict()
        sub_diff_df = diff_record[diff_record["dts_idx"]==dts_idx]
        sub_diff_df["label"] = get_label(sub_diff_df, prop_thresh)
        sub_diff_df["mode"] = [cat_dict[int(li)] if li!=0 else "norm" for li in sub_diff_df["label"].values]
        label_record = pd.concat((label_record, sub_diff_df))
    label_record.to_csv("./TEM_GAN/GAN_interpolate/diff_record.csv")

    query_record = label_record[label_record["label"]!=0]
    query_record.to_csv("./TEM_GAN/GAN_interpolate/query_record.csv")