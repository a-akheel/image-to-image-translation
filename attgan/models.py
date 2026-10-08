"""Keras port of LynnHo/AttGAN-Tensorflow module.py (MIT).

    Reference f1231fcc581133c9a7d8451e26a8bcd293fe03d5.
    SAME padding, TF-Slim BN epsilon/momentum, LeakyReLU slope and channel
    concatenation order are important for compatibility with pretrained weights.
"""
import tensorflow as tf
from tensorflow import keras
from .attributes import ATTRIBUTES


def join_attributes(x, attributes):
    shape = tf.shape(x)
    a = tf.broadcast_to(attributes[:, None, None, :],
                        [shape[0], shape[1], shape[2], tf.shape(attributes)[-1]])
    return tf.concat([x, a], axis=-1)


class Block(keras.layers.Layer):
    def __init__(self, channels, transpose=False, **kwargs):
        super().__init__(**kwargs)
        conv = keras.layers.Conv2DTranspose if transpose else keras.layers.Conv2D
        self.conv = conv(channels, 4, strides=2, padding="same", use_bias=False)
        self.norm = keras.layers.BatchNormalization(momentum=0.999, epsilon=0.001)
        self.transpose = transpose

    def call(self, x, training=False):
        x = self.norm(self.conv(x), training=training)
        return tf.nn.relu(x) if self.transpose else tf.nn.leaky_relu(x, alpha=0.2)


class Generator(keras.Model):
    def __init__(self, dim=64, n_attributes=len(ATTRIBUTES), **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_attributes = n_attributes
        self.encoder = [Block(min(dim * 2**i, 1024), name=f"encoder_{i}") for i in range(5)]
        self.decoder = [Block(min(dim * 2**(4-i), 1024), transpose=True,
                              name=f"decoder_{i}") for i in range(4)]
        # Explicit attributes also make every block visible to tf.train.Checkpoint.
        for i, block in enumerate(self.encoder + self.decoder):
            setattr(self, f"block_{i}", block)
        self.output_layer = keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same")

    def call(self, inputs, training=False):
        x, attributes = inputs
        tf.debugging.assert_equal(tf.shape(attributes)[-1], self.n_attributes)
        return self.decode(self.encode(x, training=training), attributes, training=training)

    def encode(self, x, training=False):
        features = []
        for block in self.encoder:
            x = block(x, training=training)
            features.append(x)
        return features

    def decode(self, features, attributes, training=False):
        x = join_attributes(features[-1], attributes)
        for i, block in enumerate(self.decoder):
            x = block(x, training=training)
            if i == 0:
                x = tf.concat([x, features[-2]], axis=-1)
                x = join_attributes(x, attributes)
        return tf.nn.tanh(self.output_layer(x))


class InstanceNorm(keras.layers.Layer):
    def build(self, shape):
        self.scale = self.add_weight(name="scale", shape=(shape[-1],), initializer="ones")
        self.offset = self.add_weight(name="offset", shape=(shape[-1],), initializer="zeros")

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + 1e-6) * self.scale + self.offset


class Discriminator(keras.Model):
    def __init__(self, dim=64, fc_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for i in range(5):
            block = keras.Sequential([
                keras.layers.Conv2D(min(dim * 2**i, 1024), 4, strides=2,
                                    padding="same", use_bias=False),
                InstanceNorm(), keras.layers.LeakyReLU(negative_slope=0.2),
            ])
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        self.flatten = keras.layers.Flatten()
        self.realism = keras.Sequential([keras.layers.Dense(fc_dim),
            keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Dense(1)])
        self.classifier = keras.Sequential([keras.layers.Dense(fc_dim),
            keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Dense(len(ATTRIBUTES))])

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.flatten(x)
        return self.realism(x), self.classifier(x)
