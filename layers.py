import tensorflow as tf

relu = tf.nn.relu


def lrelu(x, leak=0.2):
    '''
    lrelu
    :param x: input tensor
    :param leak: factor
    :return: tensor
    '''
    x = tf.identity(x)
    return (0.5 * (1 + leak)) * x + (0.5 * (1 - leak)) * tf.abs(x)


def norm(x):
    '''
    layer normalization
    :param x: input tensor
    :return: layer normalized tensor
    '''
    return tf.contrib.layers.layer_norm(x)


def conv2d(inputconv,
           o_d=64,
           f_h=7,
           f_w=7,
           s_h=1,
           s_w=1,
           stddev=0.02,
           padding="VALID",
           name="conv2d",
           do_norm=True,
           do_relu=True,
           relufactor=0
           ):
    '''
    convolution layer
    :param inputconv: input tensor
    :param o_d: output channels
    :param f_h: height of filter
    :param f_w: width of filter
    :param s_h: height of strides
    :param s_w: width of strides
    :param stddev: standard deviation of parameters
    :param padding: method of padding
    :param name: operation name
    :param do_norm: whether normalize
    :param do_relu: whether ReLU
    :param relufactor: factor of lrelu
    :return: tensor
    '''
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputconv,
            o_d,
            f_w,
            s_w,
            padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm: conv = norm(conv)
        if do_relu: conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        return conv


def deconv2d(inputconv,
             o_d=64,
             f_h=7,
             f_w=7,
             s_h=1,
             s_w=1,
             stddev=0.02,
             padding="VALID",
             name="deconv2d",
             do_norm=True,
             do_relu=True,
             relufactor=0
             ):
    '''
    transpose convolution layer
    :param inputconv: input tensor
    :param o_d: output channels
    :param f_h: height of filter
    :param f_w: width of filter
    :param s_h: height of strides
    :param s_w: width of strides
    :param stddev: standard deviation of parameters
    :param padding: method of padding
    :param name: operation name
    :param do_norm: whether normalize
    :param do_relu: whether ReLU
    :param relufactor: factor of lrelu
    :return: tensor
    '''
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d_transpose(
            inputconv,
            o_d,
            [f_h, f_w],
            [s_h, s_w],
            padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm: conv = norm(conv)
        if do_relu: conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        return conv
