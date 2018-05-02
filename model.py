import tensorflow as tf
from layers import conv2d
from layers import deconv2d

tanh = tf.nn.tanh
resize = tf.image.resize_images

img_layer = 3  # 图像通道

ngf = 8
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

        patch_input = tf.random_crop(inputdisc, [1, 64, 64, 3])
        for _ in range(disc_batch_size):
            patch = tf.random_crop(inputdisc, [1, 64, 64, 3])
            patch_input = tf.concat(axis=0, values=[patch_input, patch])

        o_c1 = conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False)
        o_c2 = conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "c2")
        o_c3 = conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "c3")
        o_c4 = conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "c4")
        o_c5 = conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)

        return o_c5


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
        num_blocks = 4  # 图像块个数
        imgs = [inputgen]  # 存储不同尺寸的图像块
        conv_blocks = []  # 用于存储图像块的卷积结果

        for iter in range(num_blocks):
            img_block = resize(images=inputgen, size=[H // (scale * pow(2, iter)), W // (scale * pow(2, iter))])
            imgs.append(img_block)

        for i in range(len(imgs)):
            conv_block = conv2d(imgs[i], ngf * pow(2, i), f, f, 1, 1, 0.02, "SAME", "c" + str(i + 1), do_norm=False)
            conv_blocks.append(conv_block)

        deconv_32_64 = deconv2d(conv_blocks[4], conv_blocks[3].get_shape()[-1], ks, ks, 2, 2, 0.02, "SAME", "dc1")
        tensor_32_128 = tf.concat(values=[deconv_32_64, conv_blocks[3]], axis=3)

        deconv_64_32 = deconv2d(tensor_32_128, conv_blocks[2].get_shape()[-1], ks, ks, 2, 2, 0.02, "SAME", "dc2")
        tensor_64_64 = tf.concat(values=[deconv_64_32, conv_blocks[2]], axis=3)

        deconv_128_16 = deconv2d(tensor_64_64, conv_blocks[1].get_shape()[-1], ks, ks, 2, 2, 0.02, "SAME", "dc3")
        tensor_128_32 = tf.concat(values=[deconv_128_16, conv_blocks[1]], axis=3)

        deconv_256_8 = deconv2d(tensor_128_32, conv_blocks[0].get_shape()[-1], ks, ks, 2, 2, 0.02, "SAME", "dc4")
        tensor_256_16 = tf.concat(values=[deconv_256_8, conv_blocks[0]], axis=3)

        img_256_3 = conv2d(tensor_256_16, img_layer, ks, ks, 1, 1, 0.02, "SAME", "dc5", do_relu=False)

        outputgen = tanh(img_256_3)

        return outputgen
