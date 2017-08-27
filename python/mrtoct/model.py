import tensorflow as tf

def _leaky_relu(x, leakiness=.02):
    return tf.where(tf.less(x, .0), leakiness * x, x, name='leaky_relu')

def unet(x):
    # 512 x 512 x 1 -> 256 x 256 x 64
    with tf.name_scope('encode1'):
        enc1dw = tf.layers.conv2d(x, 64, 3, 2, 'SAME',
            activation=_leaky_relu)

    # 256 x 256 x 64 -> 128 x 128 x 128
    with tf.name_scope('encode2'):
        enc2dw = tf.layers.conv2d(enc1dw, 128, 3, 2, 'SAME',
            activation=_leaky_relu)
        enc2bn = tf.layers.batch_normalization(enc2dw)

    # 128 x 128 x 128 -> 64 x 64 x 256
    with tf.name_scope('encode3'):
        enc3dw = tf.layers.conv2d(enc2bn, 256, 3, 2, 'SAME',
            activation=_leaky_relu)
        enc3bn = tf.layers.batch_normalization(enc3dw)

    # 64 x 64 x 256 -> 32 x 32 x 512
    with tf.name_scope('encode4'):
        enc4dw = tf.layers.conv2d(enc3bn, 512, 3, 2, 'SAME',
            activation=_leaky_relu)
        enc4bn = tf.layers.batch_normalization(enc4dw)

    # 32 x 32 x 512 -> 16 x 16 x 512
    with tf.name_scope('encode5'):
        enc5dw = tf.layers.conv2d(enc4bn, 512, 3, 2, 'SAME',
            activation=_leaky_relu)
        enc5bn = tf.layers.batch_normalization(enc5dw)

    # 16 x 16 x 512 -> 8 x 8 x 512
    with tf.name_scope('encode6'):
        enc6dw = tf.layers.conv2d(enc5bn, 512, 3, 2, 'SAME',
            activation=_leaky_relu)
        enc6bn = tf.layers.batch_normalization(enc6dw)

    # 8 x 8 x 512 -> 16 x 16 x 512
    with tf.name_scope('decode6'):
        dec6up = tf.layers.conv2d_transpose(enc6bn, 512, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec6bn = tf.layers.batch_normalization(dec6up)
        dec6dp = tf.layers.dropout(dec6bn)

    # 16 x 16 x 1024 -> 32 x 32 x 512
    with tf.name_scope('decode5'):
        dec5fu = tf.concat([dec6dp, enc5bn], 3)
        dec5up = tf.layers.conv2d_transpose(dec5fu, 512, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec5bn = tf.layers.batch_normalization(dec5up)
        dec5dp = tf.layers.dropout(dec5bn)

    # 32 x 32 x 1024 -> 64 x 64 x 256
    with tf.name_scope('decode4'):
        dec4fu = tf.concat([dec5dp, enc4bn], 3)
        dec4up = tf.layers.conv2d_transpose(dec4fu, 512, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec4bn = tf.layers.batch_normalization(dec4up)
        dec4dp = tf.layers.dropout(dec4bn)

    # 64 x 64 x 512 -> 128 x 128 x 128
    with tf.name_scope('decode3'):
        dec3fu = tf.concat([dec4dp, enc3bn], 3)
        dec3up = tf.layers.conv2d_transpose(dec3fu, 256, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec3bn = tf.layers.batch_normalization(dec3up)

    # 128 x 128 x 256 -> 256 x 256 x 64
    with tf.name_scope('decode2'):
        dec2fu = tf.concat([dec3bn, enc2bn], 3)
        dec2up = tf.layers.conv2d_transpose(dec2fu, 128, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec2bn = tf.layers.batch_normalization(dec2up)

    # 256 x 256 x 64 -> 512 x 512 x 1
    with tf.name_scope('decode1'):
        dec1fu = tf.concat([dec2bn, enc1dw], 3)
        dec1up = tf.layers.conv2d_transpose(dec1fu, 64, 3, 2, 'SAME',
            activation=tf.nn.relu)
        dec1bn = tf.layers.batch_normalization(dec1up)

    with tf.name_scope('final'):
        final = tf.layers.conv2d_transpose(dec1bn, 1, 3, 1, 'SAME',
            activation=tf.nn.tanh)

    return final
