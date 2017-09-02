import tensorflow as tf

from mrtoct.model.utils import leaky_relu
from mrtoct.model.utils import xavier_init

def model(x, y):
    z = tf.concat([x, y], 3)

    conv1 = tf.layers.conv2d(z, 64, 4, 2, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)

    conv2 = tf.layers.conv2d(conv1, 128, 4, 2, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)

    conv3 = tf.layers.conv2d(conv2, 256, 4, 2, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)

    conv4 = tf.layers.conv2d(conv3, 256, 4, 2, 'VALID',
        activation=leaky_relu, kernel_initializer=xavier_init)

    conv5 = tf.layers.conv2d(conv4, 256, 4, 2, 'VALID',
        activation=leaky_relu, kernel_initializer=xavier_init)

    return tf.nn.sigmoid(conv5)
