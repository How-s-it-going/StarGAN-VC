import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    """
    generator (U-Net)
    """
    def __init__(
            self, depth, num_filters, kernel_size, activation=tf.nn.leaky_relu,
            pool_size=2, num_conv=3, out_channels=1):
        """
        generator
        Args:
            depth: Model depth (num of downsampling and upsampling)
            num_filters: Num of Filters (num_filters * cur_depth)
            kernel_size: Kernel size of conv layer
            activation: Activation of conv layer
            pool_size: Pool size
            num_conv: num of conv layer in a sampling layer
            out_channels: num of Output channel
        """
        super(Generator, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_conv = num_conv
        self.down_conv = [
            [layers.Conv1D(
                filters=num_filters*(2**i), kernel_size=kernel_size, strides=1,
                padding="causal", activation=activation, dilation_rate=2**j, name=f'down_conv_{i}_{j}')
                for j in range(num_conv)]
            for i in range(depth)]
        self.up_conv = [
            [layers.Conv1D(
                filters=num_filters*(2**i), kernel_size=kernel_size, strides=1,
                padding="causal", activation=activation, dilation_rate=2**j, name=f'up_conv_{i}_{j}')
                for j in range(num_conv)]
            for i in range(depth-1, -1, -1)]
        self.bottom = [layers.Conv1D(
                filters=num_filters*(2**depth), kernel_size=kernel_size,
                padding="causal", activation=activation, name=f'bottom_conv_{j}')
                for j in range(num_conv)]
        self.out_layer = layers.Conv1D(out_channels, 1, activation=tf.nn.sigmoid, name='generator_output_layer')

    def call(self, inputs, training=None, mask=None):
        # TODO: ターゲットのベクトルをモデルに絡める
        cur = inputs[0]
        enc_out = list()

        # DownSample
        for i in range(self.depth):
            for j in range(self.num_conv):
                cur = self.down_conv[i][j](cur)
            enc_out.append(cur)
            cur = layers.MaxPooling1D(pool_size=self.pool_size)(cur)

        # Bottom
        for i in range(self.num_conv):
            cur = self.bottom[i](cur)

        # UpSample
        for i in range(self.depth):
            cur = layers.UpSampling1D(size=self.pool_size)(cur)
            cur = ConcatWithTrans([enc_out[-i-1].shape, cur.shape])([enc_out[-i-1], cur])
            for j in range(self.num_conv):
                cur = self.up_conv[i][j](cur)

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
