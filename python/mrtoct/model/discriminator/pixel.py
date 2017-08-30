import tensorflow as tf:

from mrtoct.model import leaky_relu
from mrtoct.model import xavier_init

def model(x, y):
    z = tf.concat([x, y], 3)

    conv1 = tf.contrib.layers.conv2d(z, 64, 1, 1, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)

    return conv1