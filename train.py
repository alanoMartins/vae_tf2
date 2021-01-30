import tensorflow as tf
from vae import VAE
from datasets import get_mnist, get_cifar
from gui import start_gui

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100

LATENT_SHAPE = 2
LOG_DIR = 'tensorboard'
CHECKPOINT_DIR = './checkpoints/'

# Initialize callbacks
tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_DIR,
    monitor='total_loss',
    mode='min',
    save_freq=5,
    save_best_only=True)

# Get dataset
train_dataset, test_dataset, shape = get_mnist()

vae = VAE(shape, LATENT_SHAPE)
tb_callback.set_model(vae)

vae.load_weights(CHECKPOINT_DIR)
vae.compile(optimizer="adam")
vae.fit(train_dataset, epochs=20, callbacks=[tb_callback, model_checkpoint_callback])

# Validate
vae.evaluate(test_dataset)

# Save sampes in tensorboard
predictions = vae.random_sample()
file_writer = tf.summary.create_file_writer(LOG_DIR)
with file_writer.as_default():
    tf.summary.image("Images", predictions, max_outputs=10, step=0)

# Visual experiment
start_gui(vae)
