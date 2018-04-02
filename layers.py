import tensorflow as tf

relu = tf.nn.relu


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def norm(x):
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
        relu_conv = conv
        if do_relu: relu_conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        return conv, relu_conv


def upsample(inputconv, scale=2):
    _, w, h, _ = inputconv.get_shape().as_list()
    return tf.image.resize_images(inputconv, size=[w * scale, h * scale], method=1)
