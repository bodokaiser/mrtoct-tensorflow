import tensorflow as tf

from mrtoct.model.activation import leaky_relu

def unet_down(x, num_filters, batch_norm=True):
    x = tf.layers.conv2d(x, num_filters, 3, 2, 'SAME', activation=leaky_relu)
    if batch_norm:
        x = tf.layers.batch_normalization(x)
    return x

def unet_up(x, num_filters, batch_norm=True, dropout=False):
    x = tf.layers.conv2d_transpose(x, num_filters, 3, 2, 'SAME',
        activation=tf.nn.relu)
    if batch_norm:
        x = tf.layers.batch_normalization(x)
    if dropout:
        x = tf.layers.dropout(x)
    return x

def unet(x, num_filters):
    with tf.variable_scope('encode'):
        enc1 = unet_down(x, num_filters, batch_norm=False)
        enc2 = unet_down(enc1, 2*num_filters)
        enc3 = unet_down(enc2, 4*num_filters)
        enc4 = unet_down(enc3, 8*num_filters)
        enc5 = unet_down(enc4, 8*num_filters)

    with tf.variable_scope('decode'):
        dec5 = unet_up(enc5, 8*num_filters, dropout=True)
        dec4 = unet_up(tf.concat([dec5, enc4], 3), 8*num_filters, dropout=True)
        dec3 = unet_up(tf.concat([dec4, enc3], 3), 4*num_filters)
        dec2 = unet_up(tf.concat([dec3, enc2], 3), 2*num_filters)
        dec1 = unet_up(tf.concat([dec2, enc1], 3), num_filters)

    with tf.variable_scope('final'):
        return tf.layers.conv2d_transpose(dec1, 1, 3, 1, 'SAME',
            activation=tf.nn.tanh)

def pixel(x1, x2, num_filters, reuse):
    x = tf.concat([x1, x2], 3)
    x = tf.layers.conv2d(x, num_filters, 4, 2, 'SAME',
        activation=leaky_relu, reuse=reuse, name='conv1')
    x = tf.layers.conv2d(x, 2*num_filters, 4, 2, 'SAME',
        activation=leaky_relu, reuse=reuse, name='conv2')
    x = tf.layers.conv2d(x, 4*num_filters, 4, 2, 'SAME',
        activation=leaky_relu, reuse=reuse, name='conv3')
    x = tf.layers.conv2d(x, 8*num_filters, 4, 2, 'VALID',
        activation=leaky_relu, reuse=reuse, name='conv4')
    x = tf.layers.conv2d(x, 1, 4, 1, 'VALID',
        activation=leaky_relu, reuse=reuse, name='conv5')

    return tf.nn.sigmoid(x)