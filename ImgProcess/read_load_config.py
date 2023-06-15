import sys, os, yaml
import pandas as pd
try:
    from general_lib import *
except: 
    from lib.general_lib import *


def load_config(cfg_file=None):
    with open(cfg_file, 'r') as stream:
        try:
            content = yaml.load(stream)
            print (content)
            return content
        except yaml.YAMLError as exc:
            print(exc)
            exit()

def read_config(config, type="obs"):
    assert type in ["gen", "obs"]
    config_name = config["config_name"]
    video_magnification = float(config["video_magnification"])

    process_frame = config["process_frame"]
    gray_thresh = int(process_frame["gray_thresh"])
    smooth_deg = int(process_frame["smooth_deg"])
    noise_tol = int(process_frame["noise_tol"])

    if type=="obs":
        directory = config["directory"]
        video_name = str(directory["video_name"])
        input_dir = str(directory["input_dir"]) 
        input_dir = "{0}/{1}/".format(input_dir, video_name)
        video_suffix = str(directory["video_suffix"])
        input_video = "{0}/{1}_{2}.mp4".format(input_dir, video_name, video_suffix)

        out_dir = str(directory["out_dir"])
        out_file = '{0}/{1}/'.format(out_dir, video_name)
        w_contour_suffix = directory["w_contour_suffix"]
        w_d_min_suffix = directory["w_d_min_suffix"]
        w_base_border_suffix = directory["w_base_border_suffix"]
        base_border_file = "{0}/{1}/{1}_{2}.pickle".format(out_dir, video_name, w_base_border_suffix)
        template_matching_suffix = directory["template_matching_suffix"]

        if not os.path.isfile(input_video):
            print ("Error! Input video not found.")
            quit()

        out_dir = str(directory["out_dir"])
        makedirs(out_dir)

        template_matching = config["template_matching"]
        method = template_matching["method"]
        template_crop = template_matching["template_crop"]

        left_w_lb = int(template_crop["left_width_crop_lb"])
        left_w_ub = int(template_crop["left_width_crop_ub"])
        left_h_lb = int(template_crop["left_height_crop_lb"])
        left_h_ub = int(template_crop["left_height_crop_ub"])
        right_w_lb = int(template_crop["right_width_crop_lb"])
        right_w_ub = int(template_crop["right_width_crop_ub"])
        right_h_lb = int(template_crop["right_height_crop_lb"])
        right_h_ub = int(template_crop["right_height_crop_ub"])

        base_border_detect = config["base_border_detect"]
        shifting_params_suffix = base_border_detect["shifting_params_suffix"]
        shifting_params_file = "{0}/{1}/{1}_{2}.pickle".format(out_dir, video_name, shifting_params_suffix)
        shift_xy_suffix = base_border_detect["shift_xy_suffix"]
        smoothing_factor = base_border_detect["smoothing_factor"]
        left_knot_idx = base_border_detect["left_knot_idx"]-1
        right_knot_idx = -1 * base_border_detect["right_knot_idx"]
        tolerance_width = base_border_detect["tolerance_width"]

        config_dict = {
            # Directories
            "config_name": config_name,
            "video_magnification": video_magnification, 
            "input_dir": input_dir,
            "video_name": video_name,
            "input_video": input_video,
            "out_dir": out_dir,
            "out_file": out_file,
            "w_contour_suffix": w_contour_suffix,
            "w_d_min_suffix": w_d_min_suffix,
            "w_base_border_suffix": w_base_border_suffix,
            "template_matching_suffix": template_matching_suffix,

            # Process_frame
            "gray_thresh": gray_thresh,
            "smooth_deg": smooth_deg,
            "noise_tol": noise_tol,
    
            # Template matching
            "template_matching_method": method,
            "left_w_crop": [left_w_lb, left_w_ub],
            "left_h_crop": [left_h_lb, left_h_ub],
            "right_w_crop": [right_w_lb, right_w_ub],
            "right_h_crop": [right_h_lb, right_h_ub],

            # Base border detecting
            "shifting_params_suffix": shifting_params_suffix, 
            "shifting_params_file": shifting_params_file, 
            "shift_xy_suffix": shift_xy_suffix,
            'w_base_boder_suffix': w_base_border_suffix,
            "base_border_file": base_border_file, 
            "smoothing_factor": smoothing_factor, 
            "left_knot_idx": left_knot_idx,
            "right_knot_idx": right_knot_idx,
            "tolerance_width": tolerance_width
        }

    else:
        directory = config["directory"]
        input_dir = directory["input_dir"]
        est_dir = directory["est_dir"]
        out_dir = directory["out_dir"]
        makedirs(out_dir)

        refine_image = config["refine_image"]
        fill_tolerance = float(refine_image["fill_tolerance"])
        diameter_tolerance = float(refine_image["diameter_tolerance"])
        trim_threshold = int(refine_image["trim_threshold"])
        image_size = int(refine_image["image_size"])

        config_dict = {
            # Directories
            "config_name": config_name,
            "video_magnification": video_magnification, 
            "input_dir": input_dir,
            "est_dir": est_dir,
            "out_dir": out_dir,

            # Process_frame
            "gray_thresh": gray_thresh,
            "smooth_deg": smooth_deg,
            "noise_tol": noise_tol,

            # Refine images
            "fill_tolerance": fill_tolerance,
            "diameter_tolerance": diameter_tolerance,
            "trim_threshold": trim_threshold,
            "image_size": image_size
        }

    return config_dict

































