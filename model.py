import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tf.keras import layers


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        print('tmp')

    def __call__(self, inputs, training=None, mask=None):
        print('tmp')


class Downsample2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, name_prefix='downsample2d_block_'):
        super(Downsample2DBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name_prefix + 'h1_conv')
        self.norm = tfa.layers.InstanceNormalization(
            epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.conv2d_gates = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name_prefix + 'h1_gates')
        self.norm_gates = tfa.layers.InstanceNormalization(
            epsilon=1e-6, name=name_prefix + 'h1_norm')

    def call(self, inputs, training=None, mask=None):
        pass


class GatedLinerLayer(tf.keras.Model):
    def __init__(self):
        super(GatedLinerLayer, self).__init()
        self.mul = layers.Multiply()

    def call(self, inputs, training=None, mask=None):
