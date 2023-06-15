import numpy as np
import cv2 as cv 
import pandas as pd 
import os 
import shutil
import time 
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from VarGen import VarGen

from general_lib import *
from read_load_config import *

class GenImg_process(VarGen):
    def __init__(self, **params):
        super().__init__(type="gen", **params)
        self.params = params
        self.input_dir = params["input_dir"]
        self.out_dir = params["out_dir"]
        self.save_img_dir = self.out_dir+"/refine_imgs/"
        self.est_dir = params["est_dir"]
        self.fill_tolerance = params["fill_tolerance"]
        self.diameter_tolerance = params["diameter_tolerance"]
        self.trim_threshold = params["trim_threshold"]
        self.img_size = (params["image_size"], params["image_size"])
        self.slice_dict = {}
    
    # def density_crop(self, img):
    #     if len(np.array(img.shape))==3:
    #         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     else: 
    #         gray = img.copy()
    #     _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
    #     side_idx = []
    #     for side_coef in [1, -1]:
    #         dense_ratio = 0
    #         if side_coef==1:
    #             i = 0
    #         else:
    #             i = 1
    #         while dense_ratio<1:
    #             px_y = np.where(thresh[:, side_coef*i]==0)[0]
    #             if len(px_y)<=1:
    #                 dense_ratio = 0
    #             else:
    #                 dense_ratio = len(px_y)/(px_y.max()-px_y.min()+1)
    #             i += 1
    #             if i==gray.shape[1]:
    #                 break
    #         side_idx += [i-1]
    #     crop_img = img[:, side_idx[0]:-side_idx[1]]
    #     return crop_img, np.sum(side_idx)-1

    def discard_test(self, thresh):
        discard_flag = False
        for ti in range(thresh.shape[1]):
            fill_px = np.where(thresh[:, ti]==0)[0]
            dense_ratio = len(fill_px)/(fill_px.max()-fill_px.min()+1)
            if dense_ratio<self.fill_tolerance:
                discard_flag = True
                break
        return discard_flag

    def diameter_crop(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
        col_px = np.sum(255-thresh, axis=0)/255
        side_idx = []
        for side_coef in [1, -1]:
            dia_ratio = 0
            if side_coef==1: 
                i = 0
                check_dias = col_px[:int(len(col_px)/2)]
            else:
                i = 1
                check_dias = col_px[int(len(col_px)/2):]
            while dia_ratio<self.diameter_tolerance:
                dia_ratio = col_px[side_coef*i]/np.max(check_dias)
                i += 1
            side_idx += [i-1]
        crop_img = img[:, side_idx[0]+1:-side_idx[1]]
        return crop_img, np.sum(side_idx)-1

    def local_process(self, img_file, save_img=True, job_id=0):
        img = cv.imread(img_file)
        img = cv.resize(img, self.img_size, interpolation=cv.INTER_NEAREST)
        _, _, _, contours, cnt  = process_frame(img, gray_thresh=self.gray_thresh, 
                                                smooth_deg=self.smooth_deg, noise_tol=self.noise_tol,
                                                draw_contour=0)
        if len(cnt)==2:
            srt_contours = sorted(contours, key=lambda x: len(x), reverse=True)
            cimg = np.zeros_like(img)
            cv.drawContours(cimg, srt_contours, 0, color=255, thickness=-1)
            pts = np.where(cimg == 255)[:2]
            cimg = 255*np.ones_like(img)
            for i, j in zip(pts[0], pts[1]):
                cimg[i][j] = img[i][j]
        else:
            cimg = img.copy()
        if len(cimg.shape)==3:
            gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
        else: 
            gray = cimg.copy()
        _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
        bin_thresh = (255-thresh)/255
        crop_cols = np.where(np.sum(bin_thresh, axis=0)>1)[0] 
        img_crop = img[:, crop_cols.min():crop_cols.max()+1]

        # refine_img_crop, dens_cr_size = self.density_crop(img_crop)
        refine_img_crop, dia_cr_size = self.diameter_crop(img_crop)
        discard_flag = True
        if dia_cr_size<=self.trim_threshold:
            try:
                gray = cv.cvtColor(refine_img_crop, cv.COLOR_BGR2GRAY)
                _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
                if not self.discard_test(thresh):
                    min_dist, gray_scales, diameters, pos = self.get_slices(refine_img_crop)
                    if save_img:
                        save_img = 255*np.ones_like(img)
                        fill_height = [int(0.5*(save_img.shape[0]-refine_img_crop.shape[0])), \
                                    int(0.5*(save_img.shape[0]+refine_img_crop.shape[0]))]
                        fill_width = [int(0.5*(save_img.shape[1]-refine_img_crop.shape[1])), \
                                    int(0.5*(save_img.shape[1]+refine_img_crop.shape[1]))]
                        save_img[fill_height[0]:fill_height[1], fill_width[0]:fill_width[1]] = refine_img_crop
                        cv.imwrite(self.save_img_dir+img_file.split("/")[-1], save_img)
                    discard_flag = False
                else:
                    pass
            except ValueError:
                pass
        img_name = img_file.split("/")[-1].replace(".png", "")
        if not discard_flag:
            self.slice_dict[img_name] = {}
            self.slice_dict[img_name]["min_dist"] = min_dist
            self.slice_dict[img_name]["diameters"] = diameters
            self.slice_dict[img_name]["pos"] = pos
            self.slice_dict[img_name]["gray_scale"] = gray_scales
        return discard_flag, job_id

    def parallel_process(self, gen_features=True, n_workers=6):
        if os.path.isdir(self.save_img_dir):
            shutil.rmtree(self.save_img_dir)
        makedirs(self.save_img_dir)
        img_files = np.array([file_name for file_name in os.listdir(self.input_dir) if "png" in file_name])

        start = time.time()
        update_idx = 0
        # discard_files, used_files = [], []
        with tqdm(total=100) as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as excutor:
                future = []
                for i, img_file in enumerate(img_files):
                    future.append(excutor.submit(self.local_process, self.input_dir+img_file, True, i))
                    if (i % n_workers == 0) and (i >= 1):
                        for f in future:
                            discard_flag, job_id = f.result()
                            if int(job_id/(len(img_files)/100))>update_idx:
                                pbar.update(1)
                                update_idx += 1
                        future.clear()
                for f in future:
                    discard_flag, job_id = f.result()
                    if int(job_id/(len(img_files)/100))>update_idx:
                        pbar.update(1)
                        update_idx += 1
                future.clear()	
            pbar.update(1)
        end = time.time()
        used_num = len(list(self.slice_dict.keys()))
        print(used_num, len(img_files)-used_num)
        print("Finished in {}".format(end-start))

        if gen_features:
            self.generate_variables()
    
    def est_props(self):
        for tv in ["k", "G"]:
            est_file = self.est_dir+"/krr_{}.pickle".format(tv)
            with open(est_file, "rb") as handle:
                est = pickle.load(handle)
            pred_f = est.predict(self.features)
            self.features[tv] = pred_f 

        emb_file = self.est_dir+"/mlkr.pickle"
        with open(emb_file, "rb") as handle:
                emb = pickle.load(handle)
        coords = emb.transform(self.features)
        self.features["x"] = coords[:, 0]
        self.features["y"] = coords[:, 1]

def main():
    config_file = "./input/GAN_gen_featureAxis/processing_config.yml"
    cf = load_config(config_file)
    config = read_config(cf, type="gen")
    gen_process = GenImg_process(**config)
    gen_process.parallel_process()
    gen_process.est_props()
    save_dat = gen_process.features.loc[sorted(gen_process.features.index)]
    save_dat.to_csv(gen_process.out_dir+"GAN_features.csv")

if __name__=="__main__":
    main()
