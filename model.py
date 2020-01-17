import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    """discriminator"""
    def __init__(self, num_down, num_filters, kernel_size, activation=tf.nn.leaky_relu, num_conv=3, at_size=2**20,
                 pool_size=2):
        super(Discriminator, self).__init__()
        self.first_conv = layers.Conv1D(num_filters, kernel_size*3, 1, padding="same", activation=activation)
        self.first_conv_gates = layers.Conv1D(num_filters, kernel_size*3, 1, padding="same", activation=activation)
        self.conv = [DownsampleBlock(
                filters=num_filters*(2**(i+1)), kernel_size=kernel_size, strides=1, is_discriminator=True,
                padding="causal", activation=activation, dilation_rate=2**i, name=f'down_conv_{i}')
            for i in range(num_down)]
        self.num_down = num_down
        self.out_tf = layers.Conv1D(1, 1, activation=tf.nn.sigmoid, name='discriminator_output_layer')
        self.pool_size = pool_size
        self.fc = layers.Dense(128, activation=activation)


    def call(self, inputs, training=None, mask=None):
        cur = inputs
        for j in range(self.num_down):
            for i in self.conv[j]:
                cur = i(cur)
            if j != self.num_down - 1:
                cur = layers.MaxPooling1D(pool_size=self.pool_size)

        out_tf = self.out_tf(cur)
        return out_tf


class Generator(tf.keras.Model):
    """generator (U-Net)"""
    def __init__(
            self, depth, num_filters, kernel_size, activation=tf.nn.leaky_relu,
            pool_size=2, num_conv=3, out_channels=1, num_bottom=8):
        super(Generator, self).__init__(depth, pool_size, num_conv, num_bottom)
        self.first_conv = layers.Conv1D(num_filters, kernel_size*3, 1, activation=activation, padding="same")
        self.first_conv_gates = layers.Conv1D(num_filters, kernel_size*3, 1, activation=activation, padding="same")
        self.down = [DownsampleBlock(
                filters=num_filters*(2**(i+1)), kernel_size=kernel_size, strides=2,
                padding="causal", activation=activation, dilation_rate=2**i, name=f'down_conv_{i}')
            for i in range(depth)]
        self.bottom = [BottomBlock(
                filters=num_filters*(2**(i+1)), kernel_size=kernel_size, strides=1,
                padding="causal", activation=activation, dilation_rate=2**i, name=f'up_conv_{i}_')
            for i in range(depth-1, -1, -1)]
        self.up = [UpsampleBlock(
                filters=num_filters*(2**depth), kernel_size=kernel_size, strides=1,
                padding="causal", activation=activation, name=f'bottom_conv_{j}')
                for j in range(num_bottom)]
        self.out_layer = layers.Conv1D(out_channels, kernel_size, 1,
                                       activation=tf.nn.sigmoid, name='generator_output_layer')

    def call(self, inputs, training=None, mask=None):
        # TODO: ターゲットのベクトルをモデルに絡める
        cur = inputs[0]
        at = inputs[1]

        gate = self.first_conv_gates(cur)
        cur = self.first_conv(cur)
        cur = GatedLinerUnit()([cur, gate])

        # DownSample
        for i in self.down:
            cur = i(cur)

        # Bottom
        for i in self.bottom:
            cur = i([cur, at])

        # UpSample
        for i in self.up:
            cur = i(cur)

        return self.out_layer(cur)


class ConcatWithTrans(tf.keras.Model):
    """
    Concat with transpose input[1]
    """
    def __init__(self, input_shape, axis=-1):
        super(ConcatWithTrans, self).__init__()
        self.axis = axis
        a_s = input_shape[0]
        b_s = input_shape[1]
        self.diff = abs(a_s[1] - b_s[1])

    def call(self, inputs, training=None, mask=None):
        return layers.Concatenate(self.axis)(
            [inputs[0],
             layers.ZeroPadding1D(
                 (int(self.diff/2 if self.diff % 2 == 0 else self.diff / 2 + 1), int(self.diff/2))
             )(inputs[1])])


class GatedLinerUnit(tf.keras.Model):
    def __init__(self):
        super(GatedLinerUnit, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return layers.Multiply(inputs[0], tf.sigmoid(inputs[1]))


class PixelShuffler(tf.keras.Model):
    def __init__(self, shuffle_size=2):
        super(PixelShuffler, self).__init__(shuffle_size)

    def call(self, inputs, training=None, mask=None):
        w, c = tf.shape(inputs)[1:3]
        oc = c // self.shuffle_size
        ow = w * self.shuffle_size
        return tf.reshape(inputs, [-1, ow, oc])


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.Variable(shape=[1, w_shape[-1]], initial_value=tf.random_normal_initializer(), trainable=False, name='u')

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


class DownsampleBlock(tf.keras.Model):
    def __init__(self, is_discriminator=False, **kwargs):
        super(DownsampleBlock, self).__init__(is_discriminator)
        self.conv = layers.Conv1D(**kwargs)
        self.conv_gates = layers.Conv1D(**kwargs)

    def call(self, inputs, training=None, mask=None):
        h1 = self.conv(inputs)
        h1_gates = self.conv_gates(inputs)
        h1 = spectral_norm(h1) if self.is_discriminator else layers.BatchNormalization()(h1)
        h1_gates = spectral_norm(h1) if self.is_discriminator else layers.BatchNormalization()(h1_gates)
        h1_glu = GatedLinerUnit()([h1, h1_gates])
        return h1_glu


class BottomBlock(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BottomBlock, self).__init__()
        self.conv = layers.Conv1D(**kwargs)
        self.conv_gates = layers.Conv1D(**kwargs)

    def call(self, inputs, training=None, mask=None):
        cur = self.conv(inputs[0])
        gate = self.conv_gates(inputs[0])
        c = tf.reshape(inputs[1], [-1, 1, inputs[1].shape.dims[-1].value])
        c = tf.tile(c, [1, cur.shape.dims[1].value, 1])
        cur = tf.concat([cur, c], axis=-1)
        gate = tf.concat([gate, c], axis=-1)
        return GatedLinerUnit()([cur, gate])


class UpsampleBlock(tf.keras.Model):
    def __init__(self, shuffle_size=2, **kwargs):
        super(UpsampleBlock, self).__init__(shuffle_size)
        self.conv = layers.Conv1D(**kwargs)
        self.conv_gates = layers.Conv1D(**kwargs)

    def call(self, inputs, training=None, mask=None):
        h1 = self.conv(inputs)
        h1_gates = self.conv_gates(inputs)
        h1 = PixelShuffler(self.shuffle_size)(h1)
        h1_gates = PixelShuffler(self.shuffle_size)(h1_gates)
        h1 = layers.BatchNormalization()(h1)
        h1_gates = layers.BatchNormalization()(h1_gates)
        return GatedLinerUnit()([h1, h1_gates])


def test():
    i = layers.Input([44100, 1], 16)
    g = Generator(4, 64, 256)
    d = Discriminator(4, 64, 256)
    o = d(g((i,)))
    m = tf.keras.models.Model(inputs=i, outputs=o)


if __name__ == '__main__':
    test()
