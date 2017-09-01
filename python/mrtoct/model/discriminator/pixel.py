import tensorflow as tf:

from mrtoct.model.utils import leaky_relu
from mrtoct.model.utils import xavier_init

def model(x, y):
    z = tf.concat([x, y], 3)

    conv1 = tf.contrib.layers.conv2d(z, 64, 1, 1, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)
    conv2 = tf.contrib.layers.conv2d(conv1, 128, 1, 1, 'SAME',
        kernel_initializer=xavier_init)
    bnorm = tf.contrib.layers.batch_normalization(conv2)
    conv3 = tf.contrib.layers.conv2d(leaky_relu(bnorm), 128, 1, 1, 'SAME',
        kernel_initializer=xavier_init)

    return tf.nn.sigmoid(conv3)