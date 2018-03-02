import tensorflow as tf

from mrtoct.model.cnn import unet

xavier_init = tf.contrib.layers.xavier_initializer


def _dconv(x, num_filters, data_format):
  x = tf.layers.conv2d(x, num_filters,
                       kernel_size=4,
                       strides=2,
                       padding='same',
                       data_format=data_format,
                       kernel_initializer=xavier_init())
  return tf.nn.leaky_relu(x)


def _dfinal(x, data_format):
  x = tf.layers.conv2d(x, 1,
                       kernel_size=4,
                       strides=1,
                       padding='same',
                       kernel_initializer=xavier_init())
  return tf.nn.sigmoid(x)


def discriminator_fn(y, z, data_format):
  x = tf.concat([y, z], 1 if data_format == 'channels_first' else -1)

  with tf.variable_scope('conv1'):
    x = _dconv(x, 64, data_format)
  with tf.variable_scope('conv2'):
    x = _dconv(x, 128, data_format)
  with tf.variable_scope('conv3'):
    x = _dconv(x, 256, data_format)
  with tf.variable_scope('conv4'):
    x = _dconv(x, 512, data_format)
  with tf.variable_scope('final'):
    x = _dfinal(x, data_format)

  return x


generator_fn = unet.generator_fn
