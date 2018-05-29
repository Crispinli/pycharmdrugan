import tensorflow as tf
from layers import conv2d

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
        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 3])
        o_c1 = conv2d(patch_input, ndf, f, f, 2, 2, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = conv2d(o_c1, ndf * 2, f, f, 2, 2, "SAME", "c2", relufactor=0.2)
        o_c3 = conv2d(o_c2, ndf * 4, f, f, 2, 2, "SAME", "c3", relufactor=0.2)
        o_c4 = conv2d(o_c3, ndf * 8, f, f, 1, 1, "SAME", "c4", relufactor=0.2)
        o_c5 = conv2d(o_c4, 1, f, f, 1, 1, "SAME", "c5", do_norm=False, do_relu=False)
        return o_c5


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
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, "VALID", "c2", do_relu=False)
        return tf.nn.relu(out_res + inputres)


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

        H, W = inputgen.get_shape().as_list()[1:3]  # 图像的高和宽
        scale = 2  # 图像下采样尺度
        num_blocks = 2  # 图像块个数
        imgs = [inputgen]  # 存储不同尺寸的图像块
        conv_blocks = []  # 用于存储图像块的卷积结果

        for iter in range(num_blocks):
            img_block = resize(images=inputgen, size=[H // (scale * pow(2, iter)), W // (scale * pow(2, iter))])
            imgs.append(img_block)

        for i in range(len(imgs)):
            conv_block = conv2d(imgs[i], ngf * pow(2, i), f, f, 1, 1, "SAME", "c" + str(i + 1), do_norm=False)
            conv_block = residual(conv_block, ngf * pow(2, i), "r" + str(i + 1) + "_1")
            conv_blocks.append(conv_block)

        upsample = resize(conv_blocks[2], size=conv_blocks[1].get_shape()[1:3])
        deconv = conv2d(upsample, conv_blocks[1].get_shape()[-1], ks, ks, 1, 1, "SAME", "dc3")
        tensor = tf.concat(values=[deconv, conv_blocks[1]], axis=3)

        upsample = resize(tensor, size=conv_blocks[0].get_shape()[1:3])
        deconv = conv2d(upsample, conv_blocks[0].get_shape()[-1], ks, ks, 1, 1, "SAME", "dc4")
        tensor = tf.concat(values=[deconv, conv_blocks[0]], axis=3)

        img_256_3 = conv2d(tensor, img_layer, ks, ks, 1, 1, "SAME", "dc5")
        tensor_256_6 = tf.concat(values=[img_256_3, inputgen], axis=3)
        img = conv2d(tensor_256_6, img_layer, ks, ks, 1, 1, "SAME", "dc6", do_relu=False)

        outputgen = tanh(img)

        return outputgen
