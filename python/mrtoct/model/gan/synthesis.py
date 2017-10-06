import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def generator_conv(x, kernel_size, filters, padding, activation=tf.nn.relu):
  """Creates a synthesis generator conv layer."""
  x = tf.layers.conv3d(inputs=x,
                       filters=filters,
                       kernel_size=kernel_size,
                       padding=padding,
                       kernel_initializer=xavier_init())
  x = tf.layers.batch_normalization(x)

  return activation(x)


def generator(x, params):
  """Creates a synthesis generator network."""
  for i, ks in enumerate([9, 3, 3, 3, 9, 3, 3, 7, 3]):
    with tf.variable_scope(f'conv{i}'):
      x = generator_conv(x, ks, 64 if i in [4, 5] else 32,
                         'valid' if ks == 9 else 'same')

  with tf.variable_scope('final'):
    x = generator_conv(x, 3, 1, 'same', activation=tf.nn.tanh)

  return x


def discriminator_conv(x, filters):
  """Creates a synthesis discriminator conv layer."""
  x = tf.layers.conv3d(inputs=x,
                       filters=filters,
                       kernel_size=5,
                       padding='same',
                       kernel_initializer=xavier_init())
  x = tf.layers.batch_normalization(x)
  x = tf.nn.relu(x)
  x = tf.layers.max_pooling3d(inputs=x,
                              kernel_size=3,
                              strides=1)

  return x


def discriminator_dense(x, num_filters, activation=None):
  """Creates a synthesis discriminator dense layer."""
  x = tf.layers.dense(x, num_filters)

  if activation is not None:
    x = activation(x)

  return x


def discriminator(x, params):
  """Creates a synthesis discriminator network."""
  for i, nf in enumerate([32, 64, 128, 256]):
    with tf.variable_scope(f'layer{i}'):
      x = discriminator_conv(x, nf)

  with tf.variable_scope('final'):
    for i, nf in enumerate([512, 128, 1]):
      x = discriminator_dense(x, nf, tf.nn.sigmoid if i == 2 else None)

  return x
