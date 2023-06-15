import tensorflow.keras as keras
import segmentation_models as sm

import os
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from segmentation_models.metrics import iou_score
import yaml

def process_frame(img,gray_thresh=180, smooth_deg = 5,noise_tol = 7):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #surface 
    gray = cv.GaussianBlur(gray,(smooth_deg,smooth_deg),cv.BORDER_DEFAULT)
    ret,thresh = cv.threshold(gray,gray_thresh,255,cv.THRESH_BINARY)
    #setting threshhold base on the image often 165-185
    thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE,np.ones((noise_tol,noise_tol)))
    new_img = np.zeros(gray.shape)
    new_img[thresh==255] = [0]
    new_img[thresh==0] = [1]
    return new_img

def load_data(train_path, config):
    all_video_path = os.listdir(train_path)
    all_img = []
    all_label = []

    all_video_path.remove('dataset_11278793')
    print(all_video_path)
    for p in all_video_path:
        img_path = os.listdir(os.path.join(train_path, p, 'JPEGImages'))
        img_path = sorted(img_path, key=lambda x: int(
            x.split('.')[0].split('-')[1]))
        imgs = []
        for ip in img_path:
            img = cv.imread(os.path.join(
                train_path, p, 'JPEGImages', ip))
            imgs.append(img)

        all_img.append(imgs)

        label_path = os.listdir(os.path.join(
            train_path, p, 'SegmentationClass'))
        label_path = sorted(label_path, key=lambda x: int(
            x.split('.')[0].split('-')[1]))
        lbs = []
        for lb in label_path:
            lb = np.load(os.path.join(train_path, p, 'SegmentationClass', lb))
            lbs.append(lb)
        all_label.append(lbs)

    return all_img, all_label

def fit_params(data):
    train_img, train_lb, test_img, test_lb = data
    search_range = range(165,190,5)
    score_range = []
    for thresh in search_range:
        fit_score = []
        for i in range(len(train_img)):
            new_img = process_frame(train_img[i],thresh)
            fit_score.append(iou_score(train_lb[i],new_img[...,None]).numpy())
        score_range.append(np.mean(fit_score))

    best_params = search_range[np.argmax(score_range)]

    test_score = []
    for j in range(len(test_img)):
        new_img = process_frame(train_img[i],thresh)
        test_score.append(iou_score(test_lb[i],new_img[...,None]).numpy())
    return np.mean(test_score)

def main():
    config = yaml.safe_load(open('configs/opencv.yaml'))
    # train_img, train_lb ,test_img, test_lb = load_data('train_data',config)
    all_images, all_labels = load_data('train_data', config)

    n_fold = 5
    kf = KFold(n_splits=n_fold, random_state=1234, shuffle=True)
    cv_index = []
    for i in range(len(all_labels)):
        split = [(train, test) for train, test in kf.split(all_labels[i])]
        cv_index.append(split)

    cv_times = []
    for k in range(n_fold):
        predict_score = []
        for i in range(n_fold):
            train_img = []
            train_lb = []
            test_img = []
            test_lb = []
            for j in range(len(all_labels)):
                train_img.extend(np.array(all_images[j])[cv_index[j][i][0]])
                train_lb.extend(np.array(all_labels[j])[cv_index[j][i][0]])

                test_img.extend(np.array(all_images[j])[cv_index[j][i][1]])
                test_lb.extend(np.array(all_labels[j])[cv_index[j][i][1]])
            print('fitting new fold')
            score = fit_params((train_img, train_lb, test_img, test_lb))
            predict_score.append(score)
        print(predict_score)
        cv_times.append(predict_score)

    np.save(os.path.join(config['hyper']['save_path'],
                         'opencv', 'cv_index_score.npy'), [cv_index, cv_times])


if __name__ == "__main__":
    main()

