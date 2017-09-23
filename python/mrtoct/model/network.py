import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def lrelu(x, leakiness=.02):
    return tf.where(tf.less(x, .0), leakiness * x, x, name='leaky_relu')


def dummy(x, params):
    x = tf.layers.batch_normalization(x)
    return x[:, 16:32, 16:32, 16:32, :]


def unet_encode(x, num_filters, batch_norm=True):
    x = tf.layers.conv2d(x, num_filters, 3, 2, 'SAME',
                         activation=lrelu)
    if batch_norm:
        x = tf.layers.batch_normalization(x)

    return x


def unet_decode(x, num_filters, batch_norm=True, dropout=False):
    x = tf.layers.conv2d_transpose(x, num_filters, 3, 2, 'SAME',
                                   activation=tf.nn.relu)
    if batch_norm:
        x = tf.layers.batch_normalization(x)
    if dropout:
        x = tf.layers.dropout(x)

    return x


def unet(x, params):
    nf = params.num_filters

    enc1 = unet_encode(x, nf, batch_norm=False)
    enc2 = unet_encode(enc1, 2 * nf)
    enc3 = unet_encode(enc2, 4 * nf)
    enc4 = unet_encode(enc3, 8 * nf)
    enc5 = unet_encode(enc4, 8 * nf)

    dec5 = unet_decode(enc5, 8 * nf, dropout=True)
    dec4 = unet_decode(tf.concat([dec5, enc4], 3), 8 * nf, dropout=True)
    dec3 = unet_decode(tf.concat([dec4, enc3], 3), 4 * nf)
    dec2 = unet_decode(tf.concat([dec3, enc2], 3), 2 * nf)
    dec1 = unet_decode(tf.concat([dec2, enc1], 3), nf)

    return tf.layers.conv2d_transpose(dec1, 1, 3, 1, 'SAME',
                                      activation=tf.nn.tanh)


def pix2pix(x1, x2, params):
    nf = params.num_filters

    x = tf.concat([x1, x2], 3)

    for i, s in enumerate([nf, 2 * nf, 4 * nf]):
        x = tf.layers.conv2d(x, s, 4, 2, 'SAME', name=f'conv{i}',
                             activation=lrelu)

    x = tf.layers.conv2d(x, 8 * nf, 4, 2, 'VALID',
                         name='conv3', activation=lrelu)
    x = tf.layers.conv2d(x, 1, 4, 1, 'VALID',
                         name='conv4', activation=lrelu)

    return tf.nn.sigmoid(x)


def _conv3d(x, kernel_size, num_filters, stride=1, bnorm=True, padding='SAME',
            activation=tf.nn.relu):
    x = tf.layers.conv3d(x, num_filters, kernel_size, stride, padding,
                         kernel_initializer=xavier_init())

    if bnorm:
        x = tf.layers.batch_normalization(x)
    if activation is not None:
        x = activation(x)

    return x


def synthgen(x, params):
    x = _conv3d(x, 9, 32)
    x = _conv3d(x, 3, 32)
    x = _conv3d(x, 3, 32)
    x = _conv3d(x, 3, 32)
    x = _conv3d(x, 9, 64)
    x = _conv3d(x, 3, 64)
    x = _conv3d(x, 3, 32)
    x = _conv3d(x, 7, 32)
    x = _conv3d(x, 3, 1, 1, activation=tf.nn.tanh)

    return x


def synthdisc(x1, x2, params):
    x = x1

    for i, s in enumerate([32, 64, 128, 256]):
        x = tf.layers.conv3d(x, s, 5, 1, 'SAME', name=f'conv{i}')
        x = tf.layers.batch_normalization(x, name=f'bnorm{i}')
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling3d(x, 5, 1)

    for i, s in enumerate([512, 128, 1]):
        x = tf.layers.dense(x, s, name=f'dense{i}')

    x = tf.nn.sigmoid(x)

    return x
