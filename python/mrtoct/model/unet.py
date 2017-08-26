import tensorflow as tf

slim = tf.contrib.slim

def unet(inputs):
    net = slim.conv2d(inputs, 256, [3, 3], scope='conv')

    return net(inputs)
