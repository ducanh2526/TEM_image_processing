# import yaml
import sys
import gc
import os
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import pandas as pd
import cv2 as cv 

def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()

def makedirs(file):
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

def get_out_name(video_name, suffix, side=None, file_extension='.mp4'): 
	if side: 
		out_name = '_'.join((video_name, suffix, side))+file_extension
	else: 
		out_name = '_'.join((video_name, suffix))+file_extension
	return out_name

def brief_term(term): 
	sub_term = term.split(' ')
	brief_term = '_'.join(st[:3] for st in sub_term)
	return brief_term

def close_cap(cap, out): 
	cap.release()
	out.release()
	cv.destroyAllWindows()

def get_cap_info(cap_dir): 
    cap = cv.VideoCapture(cap_dir)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap_dict = {'cap': cap, 'height':height, 'width': width, 'fps':fps, 'totalFrames': totalFrames}
    return cap_dict

def process_frame(img, gray_thresh=185, smooth_deg=5, noise_tol=7, draw_contour=3):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #surface 
    gray = cv.GaussianBlur(gray,(smooth_deg,smooth_deg),cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(gray,gray_thresh,255,cv.THRESH_BINARY)
    #setting threshhold base on the image 
    thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE, np.ones((noise_tol,noise_tol)))
    #finding contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #sort by length of t
    cnt_surface = sorted(contours, key= lambda x: len(x),reverse=True)[:2]
    cnt_surface= list(filter(lambda x: len(x)>100, cnt_surface))

    if draw_contour>0:
        cv.drawContours(img, cnt_surface, -1, (0,255,0), draw_contour) #10

    return gray, thresh, img, contours, cnt_surface

def get_contour(cnt):                
	max_y_0 = np.max(np.squeeze(cnt[0])[:,1])
	max_y_1 = np.max(np.squeeze(cnt[1])[:,1])
	if max_y_0 > max_y_1:
		cnt_0 = np.squeeze(cnt[0])
		cnt_1 = np.squeeze(cnt[1])
	else:
		cnt_0 = np.squeeze(cnt[1])
		cnt_1 = np.squeeze(cnt[0])
	return cnt_0, cnt_1