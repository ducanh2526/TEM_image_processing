import numpy as np
import pickle
import os
import cv2 as cv
import pandas as pd
from scipy.spatial.distance import cdist
import time 
import pickle
from scipy import interpolate
from sklearn.metrics import r2_score

from BaseDetect import video_names
from estimators.Estimators import Emb, KRR
# from estimators.Estimators import Emb, KRR
from read_load_config import *
from general_lib import *

var_list = ['d_min', 'D_min_alg', 'D_min', 'mean_D',
            'var_D', 'F_D', 'A_D', 'X_base_l', 'Y_base_l', 
            'A_gs', 'mean_gs', 'var_gs', 'F_gs', 'k', 'G']

class VarGen():
    def __init__(self, type="obs", **params):
        assert type in ["obs", "gen"]
        self.type = type
        self.params = params 
        if self.type=="obs":
            self.input_video = self.params["input_video"]
            self.video_name = self.params["video_name"]
            self.out_dir = self.params["out_dir"]
            self.out_file = self.params["out_file"]
            with open(self.params["shifting_params_file"], "rb") as handle:
                self.shift_xy = pickle.load(handle)
            with open(self.params["base_border_file"], "rb") as handle:
                base_border_dict = pickle.load(handle)
            self.base_border_pos = {"left":base_border_dict["left"]["base_pos"],
                                    "right":base_border_dict["right"]["base_pos"]}
       
        self.gray_thresh = self.params["gray_thresh"]
        self.smooth_deg = self.params["smooth_deg"]
        self.noise_tol = self.params["noise_tol"]
        self.video_magnification = self.params["video_magnification"]

    def read_frames(self):
        assert self.type=="obs"
        cap = cv.VideoCapture(self.input_video)
        cap_dict = get_cap_info(self.input_video)
        for frameNo in range(cap_dict['totalFrames']):
            cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
            _, frame = cap.read()
            tmp_frame = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
            frame[thresh==255] = (255, 255, 255)
            _, _, _, _, cnt = process_frame(frame, gray_thresh=self.gray_thresh, 
                                            smooth_deg=self.smooth_deg, 
                                            noise_tol=self.noise_tol, draw_contour=0)
            if len(cnt)==2:
                left_base_pos = self.base_border_pos['left'] + self.shift_xy['x_left'][frameNo] + abs(np.min(self.shift_xy['x_left'])) + 5
                right_base_pos = self.base_border_pos['right'] + self.shift_xy['x_right'][frameNo] + abs(np.min(self.shift_xy['x_right'])) + 5
                crop_frame = frame[:, left_base_pos:right_base_pos+1]
                yield frameNo, crop_frame, tmp_frame

    def get_slices(self, crop_frame, tmp_frame=None):
        tmp_frame = crop_frame.copy() if tmp_frame is None else tmp_frame
        gray = cv.cvtColor(tmp_frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, self.gray_thresh, 255, cv.THRESH_BINARY)
        x_cnt = np.arange(tmp_frame.shape[1]) if tmp_frame is None else np.arange(5, tmp_frame.shape[1]-5) 
        cnt_0, cnt_1 = [], []
        for xi in x_cnt:
            if 0 in thresh[:, xi]:
                fill_px = np.where(thresh[:, xi]==0)[0]
                cnt_0 += [[xi, fill_px.min()]]
                cnt_1 += [[xi, fill_px.max()]]
        dist = cdist(cnt_0, cnt_1)
        # pos = np.unravel_index(np.argmin(dist), dist.shape)
        min_dist = np.min(dist)

        gray = cv.cvtColor(crop_frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
        x_cnt = np.arange(crop_frame.shape[1])
        gray = 255-gray
        gray_scales, diameters, pos = [], [], []
        for x in x_cnt: 
            gray_scales += [np.sum(gray[:, x])] 
            fill_px = np.where(thresh[:, x]==0)[0]
            diameters += [fill_px.max() - fill_px.min()]
            pos += [0.5*(fill_px.max()+fill_px.min())]
        return min_dist, np.array(gray_scales), np.array(diameters), np.array(pos)

    def get_slice_params(self, out_name=None):
        frame_range = []
        slice_diameters = []
        slice_pos = []
        slice_gray_scales = []
        min_distance = []

        for frameNo, crop_frame, tmp_frame in self.read_frames():
            # crop_frame = frame[:, left_base_pos:right_base_pos+1]
            min_dist, gray_scales, diameters, pos = self.get_slices(crop_frame, tmp_frame)
            min_distance += [min_dist]
            slice_gray_scales += [gray_scales]
            slice_diameters += [diameters]
            slice_pos += [pos]
            frame_range += [frameNo]

        self.slice_dict = {}
        for i, f in enumerate(frame_range):
            self.slice_dict[f] = {}
            self.slice_dict[f]["min_dist"] = min_distance[i]
            self.slice_dict[f]["diameters"] = slice_diameters[i]
            self.slice_dict[f]["pos"] = slice_pos[i]
            self.slice_dict[f]["gray_scale"] = slice_gray_scales[i]
        if out_name is not None: 
            with open(self.out_file+out_name, 'wb') as handle:
                pickle.dump(self.slice_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_variables(self, save_csv=None):
        fr_names =  np.array(list(self.slice_dict.keys()))

        mean_D, var_D= [], []
        X_base_l = []
        Y_base_l = []
        d_min, D_min, D_min_alg= [], [], []
        A_D, A_gs = [], []
        F_D, F_gs = [], []
        mean_gs, var_gs = [], []

        # all the features are incorporated with the video magnification
        for f in fr_names: 
            diameters = self.video_magnification * self.slice_dict[f]["diameters"].ravel()
            D_min += [np.min(diameters)]
            mean_D += [np.mean(diameters)]
            var_D += [np.var(diameters)]
            F_D += [np.sum(diameters**-2)*self.video_magnification] 
            A_D += [np.sum(diameters)*self.video_magnification] #new

            pos = self.video_magnification * self.slice_dict[f]["pos"].ravel()
            X_base_l += [self.video_magnification * len(diameters)]
            Y_base_l += [pos[0] - pos[-1]]
            D_min_alg += [np.min(diameters)*np.cos(np.arctan((pos[0]-pos[-1])/ \
                        (self.video_magnification * len(diameters))))] #new
        
            gray = self.video_magnification * self.slice_dict[f]["gray_scale"].ravel()
            A_gs += [np.sum(gray)*self.video_magnification] #new
            F_gs += [np.sum(gray**-1)*self.video_magnification] #new
            mean_gs += [np.mean(gray)] #new 
            var_gs += [np.var(gray)] #new 
            d_min += [self.slice_dict[f]["min_dist"]*self.video_magnification]

        var_names = [vn for vn in var_list if not vn in ["G", "k"]]
        features_data = []
        for var_name in var_names: 
            features_data.append(np.array(eval(var_name)))

        features_data = np.array(features_data)
        features_df = pd.DataFrame(features_data.T, index=fr_names, columns=var_names)
        self.features = features_df

    def get_props(self):
        if self.type=="obs":
            raw_dat_dir = "{0}/{1}.csv".format(self.params["input_dir"], self.video_name)
            if not os.path.isfile(raw_dat_dir):
                print('Error !!!')
                print('Measurement file is not included !!!')
                exit()
            raw_dat_df = pd.read_csv(raw_dat_dir)
            G = np.array([np.mean(raw_dat_df.iloc[i:i+80, 1])  for i in range(0, len(raw_dat_df), 80)])
            k = np.array([np.max(raw_dat_df.iloc[i:i+80, 2])  for i in range(0, len(raw_dat_df), 80)])
            self.features["G"] = G[self.features.index]
            self.features["k"] = k[self.features.index]
            self.features = self.features[self.features["G"]>=0]

def get_estimators(save_csv_dir="./output/", nlfs_dir="./input/NLFS/", save_est_dir="./input/Estimators/"):
    dat_df = pd.read_csv(save_csv_dir, index_col=0)
    makedirs(save_est_dir)
    
    emb_dir = save_est_dir+"mlkr.pickle"
    nlfs_result = pd.read_csv(nlfs_dir+"All_datasets_NLFS_k.out.csv")
    if not os.path.isfile(emb_dir):
        embedding = Emb(dat_df, nlfs_result=nlfs_result, targ_var="k")
        print(embedding.pred_vars)
        embedding.fit(return_coords=False)
        with open(emb_dir, "wb") as handle:
            pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else: 
        with open(emb_dir, "rb") as handle:
            embedding = pickle.load(handle)
    trans_coords = embedding.transform(dat_df)
    dat_df["x"] = trans_coords[:, 0]
    dat_df["y"] = trans_coords[:, 1]
    dat_df.to_csv(save_csv_dir)

    for tv in ["k", "G"]:
        est_dir = save_est_dir+"krr_{}.pickle".format(tv)
        if not os.path.isfile(est_dir):
            nlfs_result = pd.read_csv(nlfs_dir+"All_datasets_NLFS_{}.out.csv".format(tv))
            est = KRR(dat_df, nlfs_result=nlfs_result, targ_var=tv)
            est.fit()
            with open(est_dir, "wb") as handle:
                pickle.dump(est, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Save at {}".format(est_dir))
        else: 
            with open(est_dir, "rb") as handle:
                est = pickle.load(handle)
        pred_prop = est.predict(dat_df)
        score = r2_score(pred_prop, dat_df[tv].values)
        print("R2 score in predicting {0}: {1}".format(tv, score))

            
def main(gen_features=True, nlfs_dir="./input/NLFS/", 
        save_features_dir="./output/exp_data/", save_est_dir="./input/Estimators/"):
    save_csv = save_features_dir+"/All_datasets_features.csv"
    if gen_features:
        all_features_df = pd.DataFrame()
        for video_idx, video_name in enumerate(video_names):
            print(video_name) 
            config_file = "./input/exp_data/{0}/{0}_config.yml".format(video_name)
            cf = load_config(config_file)
            config = read_config(cf)

            var_gen = VarGen(type="obs", **config)
            var_gen.get_slice_params()
            var_gen.generate_variables()
            var_gen.get_props()
            var_gen.features.to_csv("{0}/{1}_features.csv".format(var_gen.out_file, var_gen.video_name))
            features_df = var_gen.features.copy()
            print(len(features_df))
            features_df["fr_names"] = ["{0}_{1}".format(video_idx+1, fi) for fi in features_df.index]
            features_df = features_df.set_index("fr_names")
            all_features_df = pd.concat((all_features_df, features_df))
        all_features_df.to_csv(save_csv)
    if os.path.isdir(nlfs_dir):
        get_estimators(save_csv, nlfs_dir=nlfs_dir, save_est_dir=save_est_dir)

if __name__=="__main__":
    main()
        

