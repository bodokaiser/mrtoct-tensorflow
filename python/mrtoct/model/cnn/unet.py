import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def _encode(x, num_filters, batch_norm=True):
  x = tf.layers.conv2d(x, num_filters, 4, 2, 'same',
                       kernel_initializer=xavier_init())

  if batch_norm:
    x = tf.layers.batch_normalization(x)

  return tf.nn.leaky_relu(x)


def _decode(x, num_filters, batch_norm=True, dropout=False):
  x = tf.layers.conv2d_transpose(x, num_filters, 4, 2, padding='same',
                                 kernel_initializer=xavier_init())

  if batch_norm:
    x = tf.layers.batch_normalization(x)
  if dropout:
    x = tf.layers.dropout(x)

  return tf.nn.relu(x)


def _final(x):
  x = tf.layers.conv2d_transpose(x, 1, 4, 1, padding='same',
                                 kernel_initializer=xavier_init())
  return tf.nn.tanh(x)


def network_fn(x):
  filters = [64, 128, 256, 512, 512]

  enc = []
  dec = []

  for i, f in enumerate(filters):
    with tf.variable_scope(f'encode{i}'):
      x = x if i == 0 else enc[-1]

      enc.append(_encode(x, num_filters=f, batch_norm=i != 0))

  filters.reverse()

  for i, f in enumerate(filters):
    with tf.variable_scope(f'decode{i}'):
      x = enc[-1] if i == 0 else tf.concat([dec[-1], enc[-(i + 1)]], -1)

      dec.append(_decode(x, num_filters=f, dropout=i == 0))

  with tf.variable_scope('final'):
    return _final(dec[-1])
