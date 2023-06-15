import keras
# or from tensorflow import keras
import segmentation_models as sm

import os
import cv2
import numpy as np

import PIL.Image
import pickle
import time
import argparse

keras.backend.set_image_data_format('channels_last')

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def create_model(config):
    input_shape = (None, None, 1)
    if config['model']['encoder_weights']:
        input_shape = (None, None, 3)
    model = sm.Unet('resnet34', classes=2, input_shape=input_shape,
                    activation='softmax', encoder_weights=config['model']['encoder_weights'])

    return model

def main(args):
    config  = args.config
    model = create_model(config)
    weights = os.listdir(os.path.join(
        config['hyper']['save_path'], 'models/'))
    best_weight = sorted(weights, key=lambda x: int(
        x.split('.')[0].split('-')[1]))[-1]

    print('Loading best model: ', best_weight)
    model.load_weights(os.path.join(
        config['hyper']['save_path'], 'models', best_weight))

    # cv2.VideoWriter_fourcc('avc1')
    cap = cv2.VideoCapture(args.video)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    out = cv2.VideoWriter(args.video_save,
                    cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

    i = 0
    s = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            img = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            a = np.squeeze(model.predict(np.array([gray])))

            res = np.zeros((width, height)).astype(np.uint8)
            res[np.where(a[:, :, 1] > 0.25)] = 1

            img[np.where(res == 0)] = [255, 255, 255]
            out.write(img)

            i += 1
            if i % 100 == 0:
                print(i)
        else:
            break

    cap.release()
    out.release()
    print(time.time()-s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', type=str, help='Path to dataset configs')
    parser.add_argument('video', type=str, help='Path to video mp4 for segmentation')
    parser.add_argument('video_save', type=str, help='Path to save video mp4 after segmentation')

    args = parser.parse_args()
    main(args)
