import tensorflow as tf

from mrtoct.model import layers
from mrtoct.model.cnn import unet


def generator_network(params):
  inputs = layers.Input(shape=(None, None, 1))
  outputs = unet.generator_network(params)(inputs)

  return layers.Network(inputs, outputs, name='generator')


def discriminator_conv_layer(inputs, num_filters):
  outputs = layers.Conv2D(num_filters, 4, 2)(inputs)
  outputs = layers.LeakyReLU()(outputs)

  return outputs


def discriminator_final_layer(inputs):
  outputs = layers.Conv2D(1, 4, 1)(inputs)
  outputs = layers.Activation(tf.nn.sigmoid)(outputs)

  return outputs


def discriminator_network(params):
  inputs = outputs = layers.Input(shape=(None, None, 2))

  for i, a in enumerate([1, 2, 4, 8]):
    with tf.variable_scope(f'conv{i}'):
      outputs = discriminator_conv_layer(outputs, a * params.num_filters)

  with tf.variable_scope('final'):
    outputs = discriminator_final_layer(outputs)

  return layers.Network(inputs, outputs, name='discriminator')
