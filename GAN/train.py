import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os
import shutil

from model_pgan import PGAN, WeightedSum
from tensorflow.keras import backend

# Create a Keras callback that periodically saves generated images and updates alpha in WeightedSum layers


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=512, prefix=''):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(
            shape=[num_img, self.latent_dim], seed=9434)
        self.steps_per_epoch = 0
        self.epochs = 0
        self.steps = self.steps_per_epoch * self.epochs
        self.n_epoch = 0
        self.prefix = prefix

    def set_prefix(self, prefix=''):
        self.prefix = prefix

    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            samples = self.model.generator(self.random_latent_vectors)
            samples = (samples * 0.5) + 0.5
            n_grid = int(sqrt(self.num_img))

            fig, axes = pyplot.subplots(
                n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
            sample_grid = np.reshape(
                samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))
            x = (sample_grid[0][0] * 255).astype(np.uint8)
            # print(x, x.shape)
            for i in range(n_grid):
                for j in range(n_grid):
                    axes[i][j].set_axis_off()
                    samples_grid_i_j = Image.fromarray(
                        (sample_grid[i][j] * 255).astype(np.uint8)[..., 0])
                    samples_grid_i_j = samples_grid_i_j.resize((128, 128))
                    axes[i][j].imshow(np.array(samples_grid_i_j), cmap='gray')
            title = f'images_2020_cropnc_32/plot_{self.prefix}_{epoch:05d}.png'
            pyplot.savefig(title, bbox_inches='tight')
            print(f'\n saved {title}')
            pyplot.close(fig)

    def on_batch_begin(self, batch, logs=None):
        # Update alpha in WeightedSum layers
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / \
            float(self.steps - 1)
        # print(f'\n {self.steps}, {self.n_epoch}, {self.steps_per_epoch}, {alpha}')
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

# Normalilze [-1, 1] input images


def preprocessing_image(img):
    # img = tf.where(img >= 127, 255, 0)
    img = tf.cast(img, 'float32')
    img = (img - 127.5) / 127.5
    return img


strategy = tf.distribute.MirroredStrategy()
NUM_REP = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(NUM_REP))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

# DEFINE FILEPATH AND PARAMETERS
# can use celeb A mask dataset on https://github.com/switchablenorms/CelebAMask-HQ
DATA_ROOT = '/home/share/tem_project/crop_NCs'
NOISE_DIM = 32
# Set the number of batches, epochs and steps for trainining.
# Look 800k images(16x50x1000) per each lavel
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 8, 4]
EPOCHS = 16
STEPS_PER_EPOCH = int(2020 / NUM_REP)

train_dataset = image_dataset_from_directory(directory=DATA_ROOT,
                                             color_mode='grayscale',
                                             image_size=(4, 4),
                                             batch_size=BATCH_SIZE[0]*NUM_REP,
                                             interpolation='bicubic',
                                             labels=None)
train_dataset = train_dataset.map(preprocessing_image, num_parallel_calls=2)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.with_options(options)

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)


cbk = GANMonitor(num_img=16, latent_dim=NOISE_DIM, prefix='0_init')
cbk.set_steps(steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

with strategy.scope():
    # Instantiate the PGAN(PG-GAN) model.
    pgan = PGAN(
        latent_dim=NOISE_DIM,
        d_steps=1,
    )

    # checkpoint_path = f"ckpts_2020_cropnc_32/pgan_{cbk.prefix}.ckpt"

    # # Compile models
    # pgan.compile(
    #     d_optimizer=discriminator_optimizer,
    #     g_optimizer=generator_optimizer,
    # )

    # # Start training the initial generator and discriminator
    # pgan.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH,
    #         epochs=EPOCHS, callbacks=[cbk], verbose=2)
    # pgan.save_weights(checkpoint_path)

# shutil.copy('model_pgan.py', 'ckpts_2520_cropnc_32/model_pgan.py')
# shutil.copy('train.py', 'ckpts_2520_cropnc_32/train.py')

with strategy.scope():
    prefix = '0_init'
    pgan.load_weights(f"ckpts_2020_cropnc_32/pgan_{prefix}.ckpt")

    # # #inference
    for n_depth in range(1, 5):
        pgan.n_depth = n_depth
        prefix = f'{n_depth}_fade_in'
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()

        pgan.load_weights(f"ckpts_2020_cropnc_32/pgan_{prefix}.ckpt")

        prefix = f'{n_depth}_stabilize'
        pgan.stabilize_generator()
        pgan.stabilize_discriminator()

        pgan.load_weights(f"ckpts_2020_cropnc_32/pgan_{prefix}.ckpt")

print('Restored from ', n_depth, ' step stablize')
# Train faded-in / stabilized generators and discriminators
with strategy.scope():
    for n_depth in range(5, 7):
        # Set current level(depth)
        pgan.n_depth = n_depth

        # Set parameters like epochs, steps, batch size and image size
        steps_per_epoch = STEPS_PER_EPOCH
        epochs = int(EPOCHS*(BATCH_SIZE[0]/BATCH_SIZE[n_depth]) / 2)
        # epochs = 6
        DATA_ROOT = f'/home/share/tem_project/crop_NCs'
        train_dataset = image_dataset_from_directory(directory=DATA_ROOT,
                                                     color_mode='grayscale',
                                                     image_size=(
                                                         4*(2**n_depth), 4*(2**n_depth)),
                                                     batch_size=BATCH_SIZE[n_depth] *
                                                     NUM_REP,
                                                     interpolation='bicubic',
                                                     labels=None)
        train_dataset = train_dataset.map(
            preprocessing_image, num_parallel_calls=8)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.with_options(options)

        cbk.set_prefix(prefix=f'{n_depth}_fade_in')
        cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs)

        # Put fade in generator and discriminator
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()

        pgan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
        )

        # Train fade in generator and discriminator
        pgan.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                 epochs=epochs, callbacks=[cbk], verbose=2)
        # Save models
        checkpoint_path = f"ckpts_2520_cropnc_32/pgan_{cbk.prefix}.ckpt"
        pgan.save_weights(checkpoint_path)

        # # Change to stabilized generator and discriminator
        cbk.set_prefix(prefix=f'{n_depth}_stabilize')
        pgan.stabilize_generator()
        pgan.stabilize_discriminator()

        pgan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
        )

        # Train stabilized generator and discriminator
        pgan.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                 epochs=epochs, callbacks=[cbk], verbose=2)
        # Save models
        checkpoint_path = f"ckpts_2520_cropnc_32/pgan_{cbk.prefix}.ckpt"
        pgan.save_weights(checkpoint_path)
