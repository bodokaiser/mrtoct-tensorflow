import tensorflow as tf

slim = tf.contrib.slim

def unet(inputs):
    with slim.arg_scope([slim.conv2d]):
        net = slim.conv2d(inputs, 256, 3)
        net = slim.conv2d(net, 1, 3)

    return net
