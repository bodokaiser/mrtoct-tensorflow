import tensorflow as tf

xavier_init = tf.contrib.layers.xavier_initializer


def _gconv(x, kernel_size, filters, padding, activation=tf.nn.relu):
  """Creates a synthesis generator conv layer."""
  x = tf.layers.conv3d(inputs=x,
                       filters=filters,
                       kernel_size=kernel_size,
                       padding=padding,
                       kernel_initializer=xavier_init())
  x = tf.layers.batch_normalization(inputs=x)

  return activation(x)


def _dconv(x, filters, kernel_size=5, padding='same'):
  """Creates a synthesis discriminator conv layer."""
  x = tf.layers.conv3d(inputs=x,
                       filters=filters,
                       kernel_size=kernel_size,
                       padding=padding,
                       kernel_initializer=xavier_init())
  x = tf.layers.batch_normalization(x)
  x = tf.nn.relu(x)
  x = tf.layers.max_pooling3d(inputs=x, pool_size=3, strides=1)

  return x


def _ddense(x, units, activation=None):
  """Creates a synthesis discriminator dense layer."""
  x = tf.layers.dense(x, units=units)

  if activation is not None:
    x = activation(x)

  return x


def generator_fn(x):
  """Creates a synthesis generator network."""
  for i, ks in enumerate([9, 3, 3, 3, 9, 3, 3, 7, 3]):
    with tf.variable_scope(f'conv{i}'):
      x = _gconv(x, kernel_size=ks,
                 filters=64 if i in [4, 5] else 32,
                 padding='valid' if ks == 9 else 'same')

  with tf.variable_scope('final'):
    x = _gconv(x, kernel_size=3,
               filters=1,
               padding='same',
               activation=tf.nn.tanh)

  return x


def discriminator_fn(x):
  """Creates a synthesis discriminator network."""
  for i, nf in enumerate([32, 64, 128, 256]):
    with tf.variable_scope(f'layer{i}'):
      x = _dconv(x, filters=nf)

  with tf.variable_scope('final'):
    for i, nu in enumerate([512, 128, 1]):
      x = _ddense(x, units=nu, activation=tf.nn.sigmoid if i == 2 else None)

  return x
