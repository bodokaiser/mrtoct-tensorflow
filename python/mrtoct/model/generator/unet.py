import tensorflow as tf

from mrtoct.model.utils import xavier_init
from mrtoct.model.utils import leaky_relu

def _down(x, num_filters, batch_norm):
    layer = tf.layers.conv2d(x, num_filters, 3, 2, 'SAME',
        activation=leaky_relu, kernel_initializer=xavier_init)

    if batch_norm:
        layer = tf.layers.batch_normalization(layer)

    return layer

def _up(x, num_filters, batch_norm, dropout):
    layer = tf.layers.conv2d_transpose(x, num_filters, 3, 2, 'SAME',
            activation=tf.nn.relu, kernel_initializer=xavier_init)

    if batch_norm:
        layer = tf.layers.batch_normalization(layer)
    if dropout:
        layer = tf.layers.dropout(layer)

    return layer

def model(x):
    # 512 x 512 x 1 -> 256 x 256 x 64
    with tf.name_scope('encode1'):
        enc1 = _down(x, 64, False)

    # 256 x 256 x 64 -> 128 x 128 x 128
    with tf.name_scope('encode2'):
        enc2 = _down(enc1, 128, True)

    # 128 x 128 x 128 -> 64 x 64 x 256
    with tf.name_scope('encode3'):
        enc3 = _down(enc2, 256, True)

    # 64 x 64 x 256 -> 32 x 32 x 512
    with tf.name_scope('encode4'):
        enc4 = _down(enc3, 512, True)

    # 32 x 32 x 512 -> 16 x 16 x 512
    with tf.name_scope('encode5'):
        enc5 = _down(enc4, 512, True)

    # 16 x 16 x 1024 -> 32 x 32 x 512
    with tf.name_scope('decode5'):
        dec5 = _up(enc5, 512, True, True)

    # 32 x 32 x 1024 -> 64 x 64 x 256
    with tf.name_scope('decode4'):
        dec4 = _up(tf.concat([dec5, enc4], 3), 512, True, True)

    # 64 x 64 x 512 -> 128 x 128 x 128
    with tf.name_scope('decode3'):
        dec3 = _up(tf.concat([dec4, enc3], 3), 256, True, False)

    # 128 x 128 x 256 -> 256 x 256 x 64
    with tf.name_scope('decode2'):
        dec2 = _up(tf.concat([dec3, enc2], 3), 128, True, False)

    # 256 x 256 x 64 -> 512 x 512 x 1
    with tf.name_scope('decode1'):
        dec1 = _up(tf.concat([dec2, enc1], 3), 64, True, False)

    with tf.name_scope('final'):
        fin = tf.layers.conv2d_transpose(dec1, 1, 3, 1, 'SAME',
            activation=tf.nn.tanh, kernel_initializer=xavier_init)

    return fin