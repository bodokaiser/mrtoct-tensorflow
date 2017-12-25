import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def encode(x, num_filters, batch_norm=True):
  x = tf.layers.conv2d(x, num_filters, 4, 2, 'same',
                       kernel_initializer=xavier_init())
  if batch_norm:
    x = tf.layers.batch_normalization(x, fused=True)

  return tf.nn.leaky_relu(x)


def decode(x, num_filters, batch_norm=True, dropout=False):
  x = tf.layers.conv2d_transpose(x, num_filters, 4, 2, 'same',
                                 kernel_initializer=xavier_init())
  if batch_norm:
    x = tf.layers.batch_normalization(x, fused=True)
  if dropout:
    x = tf.layers.dropout(x)

  return tf.nn.relu(x)


def final(x):
  x = tf.layers.conv2d_transpose(x, 1, 3, 1, 'same',
                                 kernel_initializer=xavier_init())
  return tf.nn.tanh(x)


def generator_fn(x):
  with tf.variable_scope('encode1'):
    enc1 = encode(x, 64, batch_norm=False)
  with tf.variable_scope('encode2'):
    enc2 = encode(enc1, 128)
  with tf.variable_scope('encode3'):
    enc3 = encode(enc2, 256)
  with tf.variable_scope('encode4'):
    enc4 = encode(enc3, 512)
  with tf.variable_scope('encode5'):
    enc5 = encode(enc4, 512)

  with tf.variable_scope('decode5'):
    dec5 = decode(enc5, 512, dropout=True)
  with tf.variable_scope('decode4'):
    dec4 = decode(tf.concat([dec5, enc4], -1), 512, dropout=True)
  with tf.variable_scope('decode3'):
    dec3 = decode(tf.concat([dec4, enc3], -1), 256)
  with tf.variable_scope('decode2'):
    dec2 = decode(tf.concat([dec3, enc2], -1), 128)
  with tf.variable_scope('decode1'):
    dec1 = decode(tf.concat([dec2, enc1], -1), 64)

  with tf.variable_scope('final'):
    y = final(dec1)

  return y
