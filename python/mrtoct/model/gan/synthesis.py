import tensorflow as tf


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
