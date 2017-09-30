import tensorflow as tf

from mrtoct.model import layers


def generator_conv(inputs, kernel_size, num_filters, padding,
                   activation=tf.nn.relu):
  """Creates a synthesis generator conv layer."""
  outputs = layers.Conv3D(num_filters, kernel_size, padding=padding)(inputs)
  outputs = layers.BatchNorm()(outputs)
  outputs = layers.Activation(activation)(outputs)

  return outputs


def generator_network(params):
  """Creates a synthesis generator network."""
  inputs = outputs = layers.Input(shape=(32, 32, 32, 1))

  for i, ks in enumerate([9, 3, 3, 3, 9, 3, 3, 7, 3]):
    with tf.variable_scope(f'conv{i}'):
      outputs = generator_conv(outputs, ks, 64 if i in [4, 5] else 32,
                               'valid' if ks == 9 else 'same')

  with tf.variable_scope('final'):
    outputs = generator_conv(outputs, 3, 1, 'same', activation=tf.nn.tanh)

  return layers.Network(inputs, outputs, name='generator')


def discriminator_conv(inputs, num_filters):
  """Creates a synthesis discriminator conv layer."""
  outputs = layers.Conv3D(num_filters, 5)(inputs)
  outputs = layers.BatchNorm()(outputs)
  outputs = layers.Activation(tf.nn.relu)(outputs)
  outputs = layers.MaxPool3D(3, 1)(outputs)

  return outputs


def discriminator_dense(inputs, num_filters, activation=None):
  """Creates a synthesis discriminator dense layer."""
  outputs = layers.Dense(num_filters)(inputs)

  if activation is not None:
    outputs = layers.Activation(activation)(outputs)

  return outputs


def discriminator_network(params):
  """Creates a synthesis discriminator network."""
  inputs = outputs = layers.Input(shape=(16, 16, 16, 1))

  for i, nf in enumerate([32, 64, 128, 256]):
    with tf.variable_scope(f'layer{i}'):
      outputs = discriminator_conv(outputs, nf)

  with tf.variable_scope('final'):
    for i, nf in enumerate([512, 128, 1]):
      outputs = discriminator_dense(outputs, nf,
                                    tf.nn.sigmoid if i == 2 else None)

  return layers.Network(inputs, outputs, name='discriminator')
