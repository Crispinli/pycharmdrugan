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

        H, W = inputdisc.get_shape().as_list()[1:3]

        fo_c1 = conv2d(inputdisc, ndf, f, f, 2, 2, "SAME", "fc1", do_norm=False, relufactor=0.2) # [1, 128, 128, 32]
        fo_c2 = conv2d(fo_c1, ndf * 2, f, f, 2, 2, "SAME", "fc2", relufactor=0.2) # [1, 64, 64, 64]
        fo_c3 = conv2d(fo_c2, ndf * 4, f, f, 2, 2, "SAME", "fc3", relufactor=0.2) # [1, 32, 32, 128]
        fo_c4 = conv2d(fo_c3, ndf * 8, f, f, 2, 2, "SAME", "fc4", relufactor=0.2) # [1, 16, 16, 256]
        fo_c5 = conv2d(fo_c4, ndf * 16, f, f, 2, 2, "SAME", "fc5", relufactor=0.2) # [1, 8, 8, 512]
        fo_c6 = conv2d(fo_c5, 1, f, f, 1, 1, "SAME", "fc6", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        tensor = resize(images=inputdisc, size=[H//2, W//2]) # [1, 128, 128, 3]
        io_c1 = conv2d(tensor, ndf, f, f, 2, 2, "SAME", "ic1", do_norm=False, relufactor=0.2) # [1, 64, 64, 32]
        io_c2 = conv2d(io_c1, ndf * 2, f, f, 2, 2, "SAME", "ic2", relufactor=0.2) # [1, 32, 32, 64]
        io_c3 = conv2d(io_c2, ndf * 4, f, f, 2, 2, "SAME", "ic3", relufactor=0.2) # [1, 16, 16, 128]
        io_c4 = conv2d(io_c3, ndf * 8, f, f, 2, 2, "SAME", "ic4", relufactor=0.2) # [1, 8, 8, 256]
        io_c5 = conv2d(io_c4, 1, f, f, 1, 1, "SAME", "ic5", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        tensor = resize(images=inputdisc, size=[H//4, W//4]) # [1, 64, 64, 3]
        bo_c1 = conv2d(tensor, ndf, f, f, 2, 2, "SAME", "bc1", do_norm=False, relufactor=0.2) # [1, 32, 32, 32]
        bo_c2 = conv2d(bo_c1, ndf * 2, f, f, 2, 2, "SAME", "bc2", relufactor=0.2) # [1, 16, 16, 64]
        bo_c3 = conv2d(bo_c2, ndf * 4, f, f, 2, 2, "SAME", "bc3", relufactor=0.2) # [1, 8, 8, 128]
        bo_c4 = conv2d(bo_c3, 1, f, f, 1, 1, "SAME", "bc4", do_norm=False, do_relu=False) # [1, 8, 8, 1]

        output = tf.concat(axis=0, values=[fo_c6, io_c5, bo_c4])

        return output


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
            img_block = resize(images=inputgen, size=[H//(scale * pow(2, iter)), W//(scale * pow(2, iter))])
            imgs.append(img_block)

        for i in range(len(imgs)):
            conv_block = conv2d(imgs[i], ngf * pow(2, i), f, f, 1, 1, "SAME", "c" + str(i + 1), do_norm=False)
            conv_blocks.append(conv_block)

        upsample = resize(conv_blocks[4], size=conv_blocks[3].get_shape()[1:3])
        deconv = conv2d(upsample, conv_blocks[3].get_shape()[-1], ks, ks, 1, 1, "SAME", "dc1")
        tensor = tf.concat(values=[deconv, conv_blocks[3]], axis=3)

        upsample = resize(tensor, size=conv_blocks[2].get_shape()[1:3])
        deconv = conv2d(upsample, conv_blocks[2].get_shape()[-1], ks, ks, 1, 1, "SAME", "dc2")
        tensor = tf.concat(values=[deconv, conv_blocks[2]], axis=3)

        upsample = resize(tensor, size=conv_blocks[1].get_shape()[1:3])
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