import tensorflow as tf
from layers import conv2d
from layers import lrelu
from layers import deconv2d

tanh = tf.nn.tanh
resize = tf.image.resize_images

img_layer = 3  # 图像通道

ngf = 32
ndf = 32

disc_batch_size = 3


def discriminator(inputdisc, name="discriminator"):
    '''
    build the discriminator
    :param inputdisc: tensor
    :param name: operation name
    :return: tensor
    '''
    with tf.variable_scope(name):
        f = 3

        H, W = inputdisc.get_shape().as_list()[1:3]

        fo_c1 = conv2d(inputdisc, ndf, f, f, 2, 2, "SAME", "fc1", do_norm=False) # [1, 128, 128, 32]
        fo_c2 = conv2d(fo_c1, ndf * 2, f, f, 2, 2, "SAME", "fc2") # [1, 64, 64, 64]
        fo_c3 = conv2d(fo_c2, ndf * 4, f, f, 2, 2, "SAME", "fc3") # [1, 32, 32, 128]
        fo_c4 = conv2d(fo_c3, ndf * 8, f, f, 2, 2, "SAME", "fc4") # [1, 16, 16, 256]
        fo_c5 = conv2d(fo_c4, ndf * 16, f, f, 2, 2, "SAME", "fc5") # [1, 8, 8, 512]
        fo_c6 = conv2d(fo_c5, 1, f, f, 1, 1, "SAME", "fc6", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        tensor = resize(images=inputdisc, size=[H//2, W//2]) # [1, 128, 128, 3]
        io_c1 = conv2d(tensor, ndf, f, f, 2, 2, "SAME", "ic1", do_norm=False) # [1, 64, 64, 32]
        io_c2 = conv2d(io_c1, ndf * 2, f, f, 2, 2, "SAME", "ic2") # [1, 32, 32, 64]
        io_c3 = conv2d(io_c2, ndf * 4, f, f, 2, 2, "SAME", "ic3") # [1, 16, 16, 128]
        io_c4 = conv2d(io_c3, ndf * 8, f, f, 2, 2, "SAME", "ic4") # [1, 8, 8, 256]
        io_c5 = conv2d(io_c4, 1, f, f, 1, 1, "SAME", "ic5", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        tensor = resize(images=inputdisc, size=[H//4, W//4]) # [1, 64, 64, 3]
        bo_c1 = conv2d(tensor, ndf, f, f, 2, 2, "SAME", "bc1", do_norm=False) # [1, 32, 32, 32]
        bo_c2 = conv2d(bo_c1, ndf * 2, f, f, 2, 2, "SAME", "bc2") # [1, 16, 16, 64]
        bo_c3 = conv2d(bo_c2, ndf * 4, f, f, 2, 2, "SAME", "bc3") # [1, 8, 8, 128]
        bo_c4 = conv2d(bo_c3, 1, f, f, 1, 1, "SAME", "bc4", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        output = tf.concat(axis=0, values=[fo_c6, io_c5, bo_c4])

        return output


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
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, "VALID", "c1", relufactor=0.2)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, "VALID", "c2", do_relu=False)
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
        o_c1 = conv2d(pad_input, ngf, f, f, 1, 1, name="c1", relufactor=0.2)
        o_c2 = conv2d(o_c1, ngf * 2, ks, ks, 2, 2, "SAME", "c2", relufactor=0.2)
        o_c3 = conv2d(o_c2, ngf * 4, ks, ks, 2, 2, "SAME", "c3", relufactor=0.2)
        o_c4 = conv2d(o_c3, ngf * 8, ks, ks, 2, 2, "SAME", "c4", relufactor=0.2)

        o_r1 = residual(o_c4, ngf * 8, "r1")
        o_r2 = residual(o_r1, ngf * 8, "r2")
        o_r3 = residual(o_r2, ngf * 8, "r3")
        o_r4 = residual(o_r3, ngf * 8, "r4")
        o_r5 = residual(o_r4, ngf * 8, "r5")
        o_r6 = residual(o_r5, ngf * 8, "r6")
        o_r7 = residual(o_r6, ngf * 8, "r7")
        o_r8 = residual(o_r7, ngf * 8, "r8")
        o_r9 = residual(o_r8, ngf * 8, "r9")

        o_c5 = deconv2d(o_r9, ngf * 4, ks, ks, 2, 2, "SAME", "c5")
        o_c5 = tf.concat(axis=3, values=[o_c5, o_c3])

        o_c6 = deconv2d(o_c5, ngf * 2, ks, ks, 2, 2, "SAME", "c6")
        o_c6 = tf.concat(axis=3, values=[o_c6, o_c2])

        o_c7 = deconv2d(o_c6, ngf * 1, ks, ks, 2, 2, "SAME", "c7")
        o_c7 = tf.concat(axis=3, values=[o_c7, o_c1])

        o_c8 = conv2d(o_c7, img_layer, f, f, 1, 1, "SAME", "c8")
        o_c8 = tf.concat(axis=3, values=[o_c8, inputgen])
        o_c8 = conv2d(o_c8, img_layer, f, f, 1, 1, "SAME", "o_c8", do_relu=False)

        return tanh(o_c8)
