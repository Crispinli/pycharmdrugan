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


def norm(x, G=64, eps=1e-5):
    '''
    Group Normalization
    :param x: input tensor
    :param G: the number of group
    :param eps: a small float number to avoid dividing by 0
    :return: normalized tensor
    '''
    with tf.variable_scope("GroupNorm"):
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [1, 1, 1, C],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [N, H, W, C]) * gamma + beta
        return x


def conv2d(inputconv,
           o_d=64,
           f_h=7,
           f_w=7,
           s_h=1,
           s_w=1,
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
            [f_w, f_h],
            [s_w, s_h],
            padding,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
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
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm: conv = norm(conv)
        if do_relu: conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        return conv
