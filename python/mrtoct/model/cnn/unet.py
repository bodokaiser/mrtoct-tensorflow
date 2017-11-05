import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def encode(x, num_filters, batch_norm=True):
  x = tf.layers.conv2d(x, num_filters, 4, 2, 'same',
                       kernel_initializer=xavier_init())
  if batch_norm:
    x = tf.layers.batch_normalization(x)

  return tf.nn.leaky_relu(x)


def decode(x, num_filters, batch_norm=True, dropout=False):
  x = tf.layers.conv2d_transpose(x, num_filters, 4, 2, 'same',
                                 kernel_initializer=xavier_init())
  if batch_norm:
    x = tf.layers.batch_normalization(x)
  if dropout:
    x = tf.layers.dropout(x)

  return tf.nn.relu(x)
