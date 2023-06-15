import tensorflow.keras as keras
import segmentation_models as sm

import os
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
import yaml
from segmentation_models.losses import bce_dice_loss, binary_crossentropy, dice_loss
from segmentation_models.metrics import iou_score
from math import ceil
import argparse
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

keras.backend.set_image_data_format('channels_last')


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def create_model(config):
    input_shape = (None, None, 1)
    if config['model']['encoder_weights']:
        input_shape = (None, None, 3)
    model = sm.Unet('resnet34', classes=2, input_shape=input_shape,
                    activation='softmax', encoder_weights=config['model']['encoder_weights'])

    return model


def load_data(train_path, config, idx):
    all_video_path = os.listdir(train_path)
    all_img = []
    all_label = []
    fix_image_size = 1280
    print(all_video_path[idx])
    config['hyper']['save_path'] += '_' + all_video_path[idx]
    p = all_video_path[idx]

    img_path = os.listdir(os.path.join(train_path, p, 'JPEGImages'))
    img_path = sorted(img_path, key=lambda x: int(
        x.split('.')[0].split('-')[1]))
    imgs = []
    for ip in img_path:
        if config['model']['encoder_weights']:
            img = cv.imread(os.path.join(train_path, p, 'JPEGImages', ip))
            result = np.zeros((fix_image_size, fix_image_size, 3))
        else:
            img = cv.imread(os.path.join(
                train_path, p, 'JPEGImages', ip), 0)
            result = np.zeros((fix_image_size, fix_image_size))

        sh = img.shape
        result[:sh[0], :sh[1]] = img
        imgs.append(result)


    label_path = os.listdir(os.path.join(
        train_path, p, 'SegmentationClass'))
    label_path = sorted(label_path, key=lambda x: int(
        x.split('.')[0].split('-')[1]))
    lbs = []
    for lb in label_path:
        lb = np.load(os.path.join(train_path, p, 'SegmentationClass', lb))
        sh = lb.shape
        result = np.zeros((fix_image_size, fix_image_size))
        result[:sh[0], :sh[1]] = lb
        lbs.append(result)
    

    return imgs, lbs


def train(model, config, data):
    train_img, train_lb, test_img, test_lb = data

    train_img = preprocess_input(np.array(train_img)[..., None])
    test_img = preprocess_input(np.array(test_img)[..., None])
    train_lb = to_categorical(np.array(train_lb), 2)
    test_lb = to_categorical(np.array(test_lb), 2)

    model.compile(loss=bce_dice_loss,
                  optimizer=keras.optimizers.Adam(config['hyper']['lr']),
                  metrics=[iou_score])

    callbacks = []
    # if not os.path.exists(config['hyper']['save_path']):
    os.makedirs(os.path.join(
        config['hyper']['save_path'], 'models'), exist_ok=True)
    print('create trained model for ', config['hyper']['save_path'])

    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['hyper']['save_path'],
                                                                           'models/', "model-{epoch}.h5"),
                                                     monitor='val_iou_score', verbose=1, mode='max',
                                                     save_best_only=True))
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_iou_score', patience=15, mode='max')

    callbacks.append(early_stop)

    history = model.fit(train_img, train_lb, batch_size=2, epochs=30,
                        callbacks=callbacks, shuffle=True, validation_data=(test_img, test_lb))

    return history


def main(args):
    config = yaml.safe_load(open(args.config))
    all_images, all_labels = load_data('train_data', config, 2)

    model = create_model(config)
    
    print('train model  with ', len(all_images),
            'data and test with ', len(all_labels))
    print(config)
    split = int(len(all_images) * 0.8)
    history = train(model, config, (all_images[:split],
                                        all_labels[:split], all_images[split:], all_labels[split:]))

    np.save(os.path.join(config['hyper']['save_path'],
                         'models', 'train_history.npy'), history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', type=str, help='Path to dataset configs')
    args = parser.parse_args()
    main(args)
