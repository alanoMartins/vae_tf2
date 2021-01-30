import tensorflow as tf
import numpy as np


def get_dataset(data, batch_size=128, train_buf=10000, test_buf=1000):
    (train_images, _), (test_images, _) = data
    train_images = np.expand_dims(train_images, -1).astype("float32") / 255
    test_images = np.expand_dims(test_images, -1).astype("float32") / 255

    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2],
                                        train_images.shape[3]).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],
                                      test_images.shape[3]).astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_buf).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_buf).batch(batch_size)

    shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])

    return train_dataset, test_dataset, shape


def get_mnist():
    return get_dataset(tf.keras.datasets.mnist.load_data())


def get_cifar():
    return get_dataset(tf.keras.datasets.cifar10.load_data())