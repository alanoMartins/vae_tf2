import tensorflow as tf
from tensorflow import keras
from net import encoder, decoder


class VAE(keras.Model):
    def __init__(self, input_shape, latent_shape):
        super(VAE, self).__init__()
        self.latent_shape = latent_shape
        self.encoder = encoder(input_shape, latent_shape)
        self.decoder = decoder(latent_shape)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

    def random_sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, self.latent_shape))
        return self.decoder(eps)

    @tf.function
    def compute_loss(self, x, trainable=True):
        z_mean, z_var, z = self.encoder(x, trainable)
        reconstructor = self.decoder(z, trainable)
        loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, reconstructor))
        loss *= x.shape[1] * x.shape[2] * x.shape[3]
        kl_loss = 1 + z_var - tf.square(z_mean) - tf.exp(z_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        total_loss = loss + kl_loss

        return [total_loss, loss, kl_loss]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, loss, kl_loss = self.compute_loss(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"total_loss": total_loss, "kl_loss": kl_loss}

    @tf.function
    def test_step(self, data):
        total_loss, loss, kl_loss = self.compute_loss(data, trainable=False)
        return {"total_loss": total_loss, "kl_loss": kl_loss}

