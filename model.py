from layers import lrelu
from layers import conv2d
from layers import deconv2d
import tensorflow as tf

tanh = tf.nn.tanh
relu = tf.nn.relu

img_layer = 3  # 图像通道

ngf = 32
ndf = 64


def residual(inputres, dim, name="resnet"):
    '''
    residual blocks
    :param inputres: input tensor
    :param dim: output channels
    :param name: operation name
    :return: tnesor
    '''
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", relufactor=0.2)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)
        return lrelu(out_res + inputres, leak=0.2)


def generator(inputgen, name="generator"):
    '''
    build the generator
    :param inputgen: input tensor
    :param name: operation name
    :return: tensor
    '''
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1", relufactor=0.2)
        o_c2 = conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = conv2d(o_c3, ngf * 8, ks, ks, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)

        # o_r1 = residual(o_c4, ngf * 8, "r1")
        # o_r2 = residual(o_r1, ngf * 8, "r2")
        # o_r3 = residual(o_r2, ngf * 8, "r3")
        # o_r4 = residual(o_r3, ngf * 8, "r4")
        # o_r5 = residual(o_r4, ngf * 8, "r5")
        # o_r6 = residual(o_r5, ngf * 8, "r6")
        # o_r7 = residual(o_r6, ngf * 8, "r7")
        # o_r8 = residual(o_r7, ngf * 8, "r8")
        # o_r9 = residual(o_r8, ngf * 8, "r9")

        o_c5 = deconv2d(o_c4, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c5 = tf.concat(axis=3, values=[o_c5, o_c3])

        o_c6 = deconv2d(o_c5, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c6")
        o_c6 = tf.concat(axis=3, values=[o_c6, o_c2])

        o_c7 = deconv2d(o_c6, ngf * 1, ks, ks, 2, 2, 0.02, "SAME", "c7")
        o_c7 = tf.concat(axis=3, values=[o_c7, o_c1])

        o_c8 = conv2d(o_c7, img_layer, f, f, 1, 1, 0.02, "SAME", "c8")
        o_c8 = tf.concat(axis=3, values=[o_c8, inputgen])
        o_c8 = conv2d(o_c8, img_layer, f, f, 1, 1, 0.02, "SAME", "o_c8", do_relu=False)

        return tanh(o_c8)


def discriminator(inputdisc, name="discriminator"):
    '''
    build the discriminator
    :param inputdisc: tensor
    :param name: operation name
    :return: tensor
    '''
    with tf.variable_scope(name):
        f = 3
        o_c1 = conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)
        return o_c5
