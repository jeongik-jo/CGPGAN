import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp

class Dense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True, lr_scale=1.0):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.lr_scale = lr_scale

    def build(self, __input_shape):
        self.he_std = tf.sqrt(1.0 / tf.cast(__input_shape[-1], 'float32'))
        self.w = tf.Variable(tf.random.normal([__input_shape[-1], self.units]) / self.lr_scale, name=self.name + '_w')
        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')
    def call(self, __inputs):
        ftr_vecs = __inputs @ self.w * self.he_std
        if self.use_bias:
            ftr_vecs += self.b
        return self.activation(ftr_vecs * self.lr_scale)


class Conv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True, upscale=False, downscale=False):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, __input_shape):
        self.multiplier = tf.sqrt(1.0 / tf.cast(self.kernel_size * self.kernel_size * __input_shape[-1], 'float32'))
        if self.upscale:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.filters, __input_shape[-1]]), name=self.name + '_w')
            self.height = __input_shape[1]
            self.width = __input_shape[2]
        else:
            self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, __input_shape[-1], self.filters]), name=self.name + '_w')

        if self.upscale or self.downscale:
            self.blur = Blur()

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, 1, 1, self.filters]), name=self.name + '_b')

    def call(self, __inputs):
        if self.upscale:
            w = tf.pad(self.w, [[1, 1], [1, 1], [0, 0], [0, 0]])
            w = w[1:, 1:] + w[:-1, 1:] + w[1:, :-1] + w[:-1, :-1]
            ftr_maps = self.blur(tf.nn.conv2d_transpose(__inputs, w * self.multiplier,
                                                        output_shape=[hp.batch_size, self.height * 2, self.width * 2, self.filters], strides=[1, 2, 2, 1], padding='SAME'))
        elif self.downscale:
            w = tf.pad(self.w, [[1, 1], [1, 1], [0, 0], [0, 0]])
            w = w[1:, 1:] + w[:-1, 1:] + w[1:, :-1] + w[:-1, :-1]
            ftr_maps = tf.nn.conv2d(self.blur(__inputs), w * self.multiplier / 4, strides=[1, 2, 2, 1], padding='SAME')
        else:
            ftr_maps = tf.nn.conv2d(__inputs, self.w * self.multiplier, strides=[1, 1, 1, 1], padding='SAME')

        if self.use_bias:
            ftr_maps += self.b

        return self.activation(ftr_maps)


class BiasAct(kr.layers.Layer):
    def __init__(self, activation=kr.activations.linear, lr_scale=1.0):
        super().__init__()
        self.activation = activation
        self.lr_scale = lr_scale

    def build(self, __input_shape):
        if len(__input_shape) == 2:
            self.b = tf.Variable(tf.zeros([1, __input_shape[-1]]), name=self.name + '_b')
        elif len(__input_shape) == 4:
            self.b = tf.Variable(tf.zeros([1, 1, 1, __input_shape[-1]]), name=self.name + '_b')
        else:
            raise AssertionError

    def call(self, __inputs):
        return self.activation(__inputs + self.b * self.lr_scale)


class Blur(kr.layers.Layer):
    def __init__(self, upscale=False, downscale=False):
        super().__init__()
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True

    def build(self, __input_shape):
        kernel = tf.cast([1, 3, 3, 1], 'float32')
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)
        self.kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, __input_shape[-1], 1])

        if self.upscale:
            self.w = __input_shape[1]
            self.h = __input_shape[2]
            self.c = __input_shape[3]
            self.kernel = self.kernel * 4
    def call(self, __inputs):
        if self.upscale:
            __inputs = tf.pad(__inputs[:, :, tf.newaxis, :, tf.newaxis, :], [[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0]])
            __inputs = tf.reshape(__inputs, [-1, self.w * 2, self.h * 2, self.c])
            return tf.nn.depthwise_conv2d(input=__inputs, filter=self.kernel, strides=[1, 1, 1, 1], padding='SAME')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=__inputs, filter=self.kernel, strides=[1, 2, 2, 1], padding='SAME')

        else:
            return tf.nn.depthwise_conv2d(input=__inputs, filter=self.kernel, strides=[1, 1, 1, 1], padding='SAME')


activation = tf.nn.leaky_relu

class Generator(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, __input_shape):
        cnt_vec = kr.Input([hp.cnt_dim])
        ctg_vec = kr.Input([hp.ctg_dim])
        ctg_dim = tf.cast(hp.ctg_dim, 'float32')
        if ctg_dim != 1.0:
            norm_ctg_vec = (ctg_vec * ctg_dim - 1) / tf.sqrt(ctg_dim - 1)
        else:
            norm_ctg_vec = ctg_vec
        ltn_vec = tf.concat([cnt_vec, norm_ctg_vec], axis=-1)

        if hp.is_mnist:
            ftr_maps = Dense(units=7 * 7 * 256, activation=activation)(ltn_vec)
            ftr_maps = kr.layers.Reshape([7, 7, 256])(ftr_maps)
            for filters in reversed([128, 256]):
                ftr_maps = Conv2D(filters=filters, kernel_size=2, activation=activation, upscale=True)(ftr_maps)
                ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
                ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
            fake_img = Conv2D(filters=1, kernel_size=1)(ftr_maps)

            self.model = kr.Model([ctg_vec, cnt_vec], fake_img)
        else:
            ftr_vec = ltn_vec
            for _ in range(4):
                ftr_vec = Dense(units=hp.cnt_dim * 2, activation=activation)(ftr_vec)
            fake_data = Dense(units=2)(ftr_vec)
            self.model = kr.Model([ctg_vec, cnt_vec], fake_data)

    def call(self, __inputs):
        return self.model(__inputs)


class Encoder(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, __input_shape):
        if hp.is_mnist:
            ftr_maps = inp = kr.Input([28, 28, 1])
            ftr_maps = Conv2D(filters=128, kernel_size=1,  activation=activation)(ftr_maps)
            for filters in [128, 256]:
                ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
                ftr_maps = Conv2D(filters=filters, kernel_size=3, activation=activation)(ftr_maps)
                ftr_maps = Conv2D(filters=filters, kernel_size=2, activation=activation, downscale=True)(ftr_maps)
            ftr_maps = Conv2D(filters=256, kernel_size=3, activation=activation)(ftr_maps)
            ftr_vec = kr.layers.Flatten()(ftr_maps)
            ctg_vec = Dense(units=hp.ctg_dim)(ftr_vec)
            self.model = kr.Model(inp, ctg_vec)
        else:
            ftr_vec = inp = kr.Input([2])
            for i in range(4):
                ftr_vec = Dense(units=hp.cnt_dim * 2, activation=activation)(ftr_vec)
            ctg_vec = Dense(units=hp.ctg_dim)(ftr_vec)
            self.model = kr.Model(inp, ctg_vec)

    def call(self, __inputs):
        return self.model(__inputs)
