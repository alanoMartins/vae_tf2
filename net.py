import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return tf.add(z_mean, tf.multiply(tf.exp(tf.multiply(0.5, z_log_var)), epsilon))


def encoder(input_shape, latent_shape):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=(2, 2), padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_shape + latent_shape)(x)
    z_mean = tf.keras.layers.Dense(latent_shape, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_shape, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    model = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    print(model.summary())
    return model


def decoder(latent_dim):
    image_channels = 1
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    # Imagem 28 x 28, colocar 7, imagem 32 x32, colocar 8
    x = tf.keras.layers.Dense(7 * 7 * 28, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 28))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=(2, 2), padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(image_channels, 3, activation="sigmoid", padding="same")(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(32 * 32 * 3,  activation="sigmoid")(x)
    # decoder_outputs = tf.keras.layers.Reshape((32, 32, 3))(x)
    model = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    print(model.summary())
    return model

