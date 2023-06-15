import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
# from scipy.spatial.distance import cdist
import cv2 as cv
import pandas as pd
import time 
import pickle
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler

from read_load_config import *
from general_lib import *

video_names = ["11278793", "1423452", "12107372", "12101805", "12094593"]
class BaseDetect():
    def __init__(self, **params):
        self.params = params 
        self.input_video = self.params["input_video"]
        self.video_name = self.params["video_name"]
        self.out_file = self.params["out_file"]
        makedirs(self.out_file)

        self.gray_thresh = self.params["gray_thresh"]
        self.smooth_deg = self.params["smooth_deg"]
        self.noise_tol = self.params["noise_tol"]

        self.left_knot_idx = self.params["left_knot_idx"]
        self.left_knot_idx = self.params["left_knot_idx"]
        
        self.variance_dict = {"left":{}, "right":{}}
        self.shift_xy = {}
    
    def template_matching(self, frameNo, template=None, w_crop=[100,250], h_crop=[50,300]):
        # input_video = self.params["input_video"]
        # video_name = self.params["video_name"]
        cap = cv.VideoCapture(self.input_video)
        cap.set(cv.CAP_PROP_POS_FRAMES,frameNo)

        _, frame = cap.read()
        method = eval(self.params["template_matching_method"])
        _, blur_frame, frame, _, _ = process_frame(frame, gray_thresh=self.gray_thresh, smooth_deg=self.smooth_deg, 
                                                    noise_tol=self.noise_tol, draw_contour=10)
        if template is None: 
            template = blur_frame[h_crop[0]:h_crop[1],w_crop[0]:w_crop[1]]
        w, h = template.shape[1::-1]
        res = cv.matchTemplate(blur_frame, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h) 
        cv.rectangle(frame, top_left, bottom_right, 255, 2)

        return frame, top_left, template
    
    def get_shift_xy(self, side='left', frame_range=None, video_enable=True):
        start = time.time()
        print('Finding shifting parameters of {} tip ...'.format(side))

        out_name = get_out_name(self.video_name, self.params["template_matching_suffix"], side) 
        cap_dict = get_cap_info(self.input_video)
        if video_enable: 
            makedirs(self.out_file)
            out = cv.VideoWriter(self.out_file+out_name, cv.VideoWriter_fourcc(*'MP4V'),
                                cap_dict['fps'],(cap_dict['width'],cap_dict['height']))
        frames = []
        pre_shift_xy = []
        updated_template  = None
        cap = cv.VideoCapture(self.input_video)
        if frame_range is None: 
            frame_range = np.arange(cap_dict['totalFrames'])
        for i, frameNo in enumerate(frame_range):
            frame, shift, template = self.template_matching(frameNo, updated_template,
                                                            w_crop=self.params[side+"_w_crop"], 
                                                            h_crop=self.params[side+"_h_crop"])
            if i==0:
                updated_template = template 
            pre_shift_xy.append(shift)
            if len(frame_range)<=5: 
                frames.append(frame)
            if video_enable:
                out.write(frame) 
        if video_enable: 
            close_cap(cap,out)

        end = time.time()
        print('Processing time: {}'.format(end-start))
        print('=======================================================')

        pre_shift_xy = np.array(pre_shift_xy)
        self.shift_xy["x_{}".format(side)] = pre_shift_xy[:, 0] - pre_shift_xy[0][0]
        self.shift_xy["y_{}".format(side)] = pre_shift_xy[:, 1] - pre_shift_xy[0][1]
        if len(frames)>0:
            return frames

    def sampling_plot(self, frame_range, tip_side="left"):
        fig = plt.figure(figsize=(5*len(frame_range),4), constrained_layout=True)
        frames = self.get_shift_xy(side=tip_side, frame_range=frame_range)
        for i in range(len(frame_range)): 
            plt.subplot(1, len(frame_range), i+1)
            plt.imshow(frames[i])
        plt.tight_layout()   
        plt.savefig('{0}/Template_matching_{1}.pdf'.format(self.out_file, tip_side))
        release_mem(fig)

    def fix_tip(self, side='left', video_enable=False):
        start = time.time()
        print('Fixing the {} tip as a reference ...'.format(side))

        input_video = self.params["input_video"]
        video_name = self.params["video_name"]
        cap = cv.VideoCapture(input_video)
        cap_dict = get_cap_info(input_video)
        height = cap_dict["height"]
        width = cap_dict["width"]

        lim_w_crop = [abs(np.min(self.shift_xy['x_'+side]))+5, width-np.max(self.shift_xy['x_'+side])-5]
        lim_h_crop = [abs(np.min(self.shift_xy['y_'+side]))+5, height-np.max(self.shift_xy['y_'+side])-5]
        w_crop = np.array(lim_w_crop) 
        h_crop = np.array(lim_h_crop)

        # out_file = self.params["out_file"]
        out_suffix = self.params["shift_xy_suffix"]
        out_name = get_out_name(video_name, out_suffix, side) 
        if video_enable: 
            makedirs(self.out_file)
            out = cv.VideoWriter(self.out_file+out_name, cv.VideoWriter_fourcc(*'MP4V'),
                                cap_dict['fps'],(w_crop[1]-w_crop[0],h_crop[1]-h_crop[0]))
  
        for frameNo in range(cap_dict["totalFrames"]):
            cap.set(cv.CAP_PROP_POS_FRAMES,frameNo)
            _, frame = cap.read()
            _, _, _, cnt, _ = process_frame(frame, gray_thresh=self.params['gray_thresh'], smooth_deg=self.params["smooth_deg"],
                                        noise_tol=self.params["noise_tol"], draw_contour=3)
            x_crop = (w_crop+self.shift_xy['x_'+side][frameNo]).astype(np.int64)
            y_crop = (h_crop+self.shift_xy['y_'+side][frameNo]).astype(np.int64)
            crop_frame = frame[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
            if video_enable: 
                out.write(crop_frame)
        
        if video_enable: 
            close_cap(cap, out)
        
        end = time.time()
        print('Processing time: {}'.format(end-start))
        print('=======================================================')

    def find_interpolate(contour, limx=(300,500), int_kind='linear'):
        ref_cnt = list(filter(lambda x: (x[0]>=limx[0])and(x[0]<=limx[1]), contour))
        ref_cnt= np.array(ref_cnt)
        
        srt_ref_cnt = sorted(ref_cnt, key= lambda x: x[0], reverse=False)
        srt_ref_cnt = np.array(srt_ref_cnt)

        x = srt_ref_cnt[:,0]
        y = srt_ref_cnt[:,1]
        
        #remove duplicated
        new_x, new_y = [],[]
        for i in np.arange(1,len(x),1):
            #print(i)
            if x[i-1]!=x[i]: 
                if abs(y[i-1]-y[i])<=3:
                    new_x.append(x[i])
                    new_y.append(y[i])
        new_x = np.array(new_x)
        new_y = np.array(new_y)
        x_ref = np.arange(limx[0],limx[1],1)
        f = interpolate.interp1d(new_x, new_y, kind = int_kind, fill_value='extrapolate')
        y_ref = f(x_ref)
        
        return x_ref, y_ref

    def get_profile_variance(self, cap_file, tip_side="left"):
        # cap_file using video with fixed tip 
        cap = cv.VideoCapture(cap_file)
        cap_dict = get_cap_info(cap_file)

        contact_areas = []
        for frameNo in range(cap_dict['totalFrames']):
            cap.set(cv.CAP_PROP_POS_FRAMES,frameNo)
            ret, frame = cap.read()
            _, binary_frame, _, _, _ = process_frame(frame,gray_thresh=200, smooth_deg=5,
                                                    noise_tol=7, draw_contour=0)                                   
            binary_frame[binary_frame==0] = 1
            binary_frame[binary_frame==255] = 0
            contact_area_x = np.sum(binary_frame, axis=0)
            contact_areas.append(contact_area_x)

        contact_areas = np.array(contact_areas)
        width_range = np.arange(0, cap_dict["width"], 1)

        variance_list = []
        for w in width_range: 
            numerator = np.var(contact_areas[:,w]) 
            denominator = np.mean(contact_areas[:,w])
            d = (numerator)/(denominator)
            variance_list.append(d)
        variance_list = np.array(variance_list)
        self.variance_dict[tip_side]["variance_list"] = variance_list

    def base_border_sampling_plot(self, smooth_variance_list, cap_dir, base_pos, knots=None, 
                                 frameNo=None, tolerance_width=15, save_file=None): 
        xs = np.arange(len(smooth_variance_list))
        ys = np.array(smooth_variance_list)

        n_col = 16
        n_row = 8
        fig_dpi = 100

        fig = plt.figure(figsize=(n_col, n_row), dpi=fig_dpi)
        # canvas = FigureCanvas(fig)
        grid_width_ratio = [1 for i in range(n_col)]
        grid = plt.GridSpec(n_row, n_col, width_ratios=grid_width_ratio)
        grid.update(wspace=0.2, hspace=0.2)

        cap_dict = get_cap_info(cap_dir)
        cap = cv.VideoCapture(cap_dir)
        if frameNo is None: 
            frameNo = cap_dict["totalFrames"]-1

        img_ax = fig.add_subplot(grid[:int(n_row), :int(n_col/2)])
        cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
        ret, frame = cap.read()
        blk = np.zeros(frame.shape, np.uint8)
        # Draw rectangles
        cv.rectangle(blk, (base_pos-tolerance_width, 0), (base_pos+tolerance_width, cap_dict["height"]), 
                        (255, 0, 0), cv.FILLED)
        # Generate result by blending both images (opacity of rectangle image is 0.3 = 30 %)
        frame = cv.addWeighted(frame, 1, blk, 0.3, 1)
        img_ax.margins(0, 0)
        img_ax.imshow(frame)
        img_ax.xaxis.set_major_locator(plt.NullLocator())
        img_ax.yaxis.set_major_locator(plt.NullLocator())
        img_ax.autoscale_view('tight')

        plot_ax = fig.add_subplot(grid[:int(n_row), int(n_col/2):])
        plot_ax.plot(xs, ys, 'b', lw=1.5)
        # plt.axis('off')

        annotate_font = {'fontname': 'serif', 'size': 6}
        if knots is not None: 
            plot_ax.scatter(xs[knots], ys[knots], color='r', s=40, marker='o')
            # for knot in knots:
            #     plt.annotate(str(knot), (0.95*xs[knot], 1.05*ys[knot]), **annotate_font)

        plt.axvline(x=base_pos, c='r', linewidth= 2.5, alpha=1.)
        # plt.xlim(np.min(xs), np.max(xs))
        plt.yticks([])
        plt.xticks([])
        grid.tight_layout(fig)

        if save_file is not None: 
            plt.savefig(save_file, transparent=True, dpi=1000)
            print('Save at: {}'.format(save_file))
        
        return fig

    def detect_base_border(self, tip_side='left', sampling_frameNo=None, cap_dir=None, save_plot_at=None): 
        variance_list = self.variance_dict[tip_side]["variance_list"]
        width_range = np.arange(len(variance_list))
        spl = UnivariateSpline(width_range, variance_list, k=1) #add k=1
        xs = width_range
        spl.set_smoothing_factor(self.params["smoothing_factor"])
        ys = spl(xs)

        knots = spl.get_knots().astype(np.int64)
        base_border_pos = knots[self.params[tip_side+'_knot_idx']]

        if sampling_frameNo is not None: 
            assert not cap_dir is None
            fig = self.base_border_sampling_plot(ys, cap_dir, base_pos=base_border_pos, 
                                                knots=knots, frameNo=sampling_frameNo, save_file=save_plot_at)
            plot.release_mem(fig)
        
        self.variance_dict[tip_side]["knots"] = knots 
        self.variance_dict[tip_side]["base_pos"] = base_border_pos 
        print("Finding {0} base at: {1}".format(tip_side, base_border_pos))

    def get_base_border(self, w_contour=True, save_shift_xy=True, save_base_pos=True): 
        cap_dict = get_cap_info(self.input_video)
        cap = cap_dict["cap"]
        width = cap_dict["width"]
        height = cap_dict["height"]
        
        w_border_name = "{0}_{1}.mp4".format(self.video_name, self.params["w_base_border_suffix"])
        if w_contour:
            w_border_name = w_border_name.split(".")[0]+"noCnt"+".mp4"
        out_dir = self.out_file+w_border_name
        out = cv.VideoWriter(out_dir, cv.VideoWriter_fourcc(*'MP4V'), cap_dict["fps"],(width, height)) 
        base_pos_df = pd.DataFrame(columns=["left", "right"])
        for frameNo in range(cap_dict['totalFrames']):
            cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
            ret, frame = cap.read()
            pos = []
            for side in ['left','right']: 
                side_pos = self.variance_dict[side]["base_pos"]+ self.shift_xy['x_'+side][frameNo] + abs(np.min(self.shift_xy['x_'+side])) + 5
                # side_pos = base_border_pos[side] + shift_xy['x_'+side][frameNo] + abs(np.min(shift_xy['x_'+side])) + 5
                pos += [side_pos]
                base_pos_df.loc["f{}".format(frameNo), side] = side_pos
            x_range = np.arange(0, frame.shape[0], 1)
            mask_scale_raw = 1/(1+(np.exp((pos[0]-x_range)/1))) + 1 - 1/(1+(np.exp((pos[1]-x_range)/1)))
            mask_scale = MinMaxScaler().fit_transform(mask_scale_raw.reshape(-1, 1))
            mask_scale = np.array([[int(a), 0, 0] for a in 255*(mask_scale)])
            mask = np.zeros(frame.shape, np.uint8) + mask_scale
            mask = mask.astype(np.uint8)
            frame = cv.addWeighted(frame, 1, mask, 1, 1)
            if w_contour:
                _, _, _, cnt, _ = process_frame(frame, gray_thresh=self.gray_thresh)
            frame[:, :, [0, 2]] = frame[:, :, [2, 0]]
            print(frameNo)
            out.write(frame)
        close_cap(cap, out)
        if save_shift_xy:
            with open(self.params["shifting_params_file"], 'wb') as handle:
                pickle.dump(self.shift_xy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if save_base_pos:
            base_pos_df.to_csv("{0}/{1}_base_pos.csv".format(self.out_file, self.video_name), index=True)
            with open(self.params["base_border_file"], 'wb') as handle:
                pickle.dump(self.variance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def test(cfg_dir="./input/", check_template_matching=True, save_fix_tip=True):
    for video_name in video_names: 
        print(video_name)
        config_file = cfg_dir+"{0}/{0}_config.yml".format(video_name)
        cf = load_config(config_file)
        config = read_config(cf)

        base_detect = BaseDetect(**config)
        cap_dict = get_cap_info(base_detect.input_video)
        for tip_side in ["left", "right"]:
            out_suffix = config["shift_xy_suffix"]
            out_name = base_detect.out_file + get_out_name(video_name, out_suffix, side=tip_side) 

            if check_template_matching:
                frame_range = np.array(np.linspace(0, 1, num=5)*cap_dict['totalFrames']-1).astype(np.int64)
                base_detect.sampling_plot(frame_range, tip_side=tip_side)
            # if (not os.path.isfile(out_name)) or (save_fix_tip is True):
            base_detect.get_shift_xy(side=tip_side)
            base_detect.fix_tip(side=tip_side, video_enable=save_fix_tip)
            base_detect.get_profile_variance(out_name, tip_side=tip_side)
            base_detect.detect_base_border(cap_dir=out_name, tip_side=tip_side, sampling_frameNo=50,
                                          save_plot_at='{0}/Static_degree_{1}_tip.pdf'.format(base_detect.out_file, tip_side))
            print("{0} base border at: {1}".format(tip_side, base_detect.variance_dict[tip_side]["base_pos"]))

def main(cfg_dir="./input/", save_base_border_pos=True, save_shifting_params=True):
    for video_name in video_names:
        print(video_name)
        config_file = cfg_dir+"{0}/{0}_config.yml".format(video_name.split("_")[0])
        cf = load_config(config_file)
        config = read_config(cf)

        start_time = time.time()
        print('Aligning bases ...')
        base_detect = BaseDetect(**config)
        for side in ['left','right']: 
            base_detect.get_shift_xy(side=side, video_enable=True)
            base_detect.fix_tip(side=side, video_enable=True)
            shift_xy_name = base_detect.out_file + get_out_name(video_name, base_detect.params["shift_xy_suffix"], side=side) 
            base_detect.get_profile_variance(shift_xy_name, tip_side=side)
            base_detect.detect_base_border(cap_dir=shift_xy_name, tip_side=side, sampling_frameNo=50,
                                          save_plot_at='{0}/Static_degree_{1}_tip.pdf'.format(base_detect.out_file, side))
        print('Finding base borders of the contact ...')
        base_detect.get_base_border(w_contour=True, save_shift_xy=save_shifting_params, save_base_pos=save_base_border_pos)                             
        end_time = time.time()
        print('Total processing time: {}'.format(end_time-start_time))

        if config_file=="TEM_config.yml":
            original_config = eval("r'{}'".format(config_file))
            copy_config = eval("r'{0}{1}_config.yml'".format(base_detect.out_file, video_name))
            shutil.copyfile(original_config, copy_config)

if __name__=="__main__":
    # test()
    main()

    





    
