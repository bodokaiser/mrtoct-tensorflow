import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def data_format_to_axis(data_format):
  return 1 if data_format == 'channels_first' else -1


def encode(x, num_filters, data_format, batch_norm=True,):
  x = tf.layers.conv2d(x, num_filters,
                       kernel_size=4,
                       strides=2,
                       padding='same',
                       data_format=data_format,
                       kernel_initializer=xavier_init())

  if batch_norm:
    x = tf.layers.batch_normalization(x, fused=True,
                                      axis=data_format_to_axis(data_format))

  return tf.nn.leaky_relu(x)


def decode(x, num_filters, data_format, batch_norm=True, dropout=False):
  x = tf.layers.conv2d_transpose(x, num_filters,
                                 kernel_size=4,
                                 strides=2,
                                 padding='same',
                                 data_format=data_format,
                                 kernel_initializer=xavier_init())
  if batch_norm:
    x = tf.layers.batch_normalization(x, fused=True,
                                      axis=data_format_to_axis(data_format))
  if dropout:
    x = tf.layers.dropout(x)

  return tf.nn.relu(x)


def final(x, data_format):
  x = tf.layers.conv2d_transpose(x, 1,
                                 kernel_size=3,
                                 strides=1,
                                 padding='same',
                                 data_format=data_format,
                                 kernel_initializer=xavier_init())
  return tf.nn.tanh(x)


def generator_fn(x, data_format):
  axis = data_format_to_axis(data_format)

  with tf.variable_scope('encode1'):
    enc1 = encode(x, 64, data_format, batch_norm=False)
  with tf.variable_scope('encode2'):
    enc2 = encode(enc1, 128, data_format)
  with tf.variable_scope('encode3'):
    enc3 = encode(enc2, 256, data_format)
  with tf.variable_scope('encode4'):
    enc4 = encode(enc3, 512, data_format)
  with tf.variable_scope('encode5'):
    enc5 = encode(enc4, 512, data_format)

  with tf.variable_scope('decode5'):
    dec5 = decode(enc5, 512, data_format, dropout=True)
  with tf.variable_scope('decode4'):
    dec4 = decode(tf.concat([dec5, enc4], axis),
                  512, data_format, dropout=True)
  with tf.variable_scope('decode3'):
    dec3 = decode(tf.concat([dec4, enc3], axis), 256, data_format)
  with tf.variable_scope('decode2'):
    dec2 = decode(tf.concat([dec3, enc2], axis), 128, data_format)
  with tf.variable_scope('decode1'):
    dec1 = decode(tf.concat([dec2, enc1], axis), 64, data_format)

  with tf.variable_scope('final'):
    y = final(dec1, data_format=data_format)

  return y
