import tensorflow as tf
from layers import conv2d
from layers import deconv2d

tanh = tf.nn.tanh
resize_image = tf.image.resize_images

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
            patch_input = tf.concat(axis=0, values=[patch_input, tf.random_crop(inputdisc, [1, 64, 64, 3])])
        o_c1 = conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
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
        f=7
        ks = 3
        _, H, W, _ = inputgen.get_shape().as_list()
        scale = 2
        imgs = [inputgen]
        conv_blocks = []

        for iter in range(4):
            imgs.append(resize_image(images=inputgen, size=[H // (scale * pow(2, iter)), W // (scale * pow(2, iter))]))
        for i in range(len(imgs)):
            conv_blocks.append(conv2d(imgs[i], ngf * pow(2, i), f, f, 1, 1, 0.02, "SAME", name="c"+str(i+1)))

        deconv_32_256 = deconv2d(conv_blocks[4], 256, ks, ks, 2, 2, 0.02, "SAME", "dc1")
        tensor_32_512 = tf.concat(axis=3, values=[deconv_32_256, conv_blocks[3]])

        deconv_64_128 = deconv2d(tensor_32_512, 128, ks, ks, 2, 2, 0.02, "SAME", "dc2")
        tensor_64_256 = tf.concat(axis=3, values=[deconv_64_128, conv_blocks[2]])

        deconv_128_64 = deconv2d(tensor_64_256, 64, ks, ks, 2, 2, 0.02, "SAME", "dc3")
        tensor_128_128 = tf.concat(axis=3, values=[deconv_128_64, conv_blocks[1]])

        deconv_256_32 = deconv2d(tensor_128_128, 32, ks, ks, 2, 2, 0.02, "SAME", "dc4")
        tensor_256_64 = tf.concat(axis=3, values=[deconv_256_32, conv_blocks[0]])

        img_256_3 = conv2d(tensor_256_64, img_layer, ks, ks, 1, 1, 0.02, "SAME", "dc5", do_relu=False)

        outputgen = tanh(img_256_3)

        return outputgen
