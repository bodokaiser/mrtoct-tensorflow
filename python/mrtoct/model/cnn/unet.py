import tensorflow as tf

from mrtoct.model import layers


def encoder_layer(inputs, num_filters, batch_norm=True):
  """Creates u-net encoder layer.

  Args:
    inputs: input tensor of shape [batch, height, width, channels]
    num_filters: number of output channels
    batch_norm=`True`: use batch normalization after convolution
  Returns:
    outputs: output tensor of shape [batch, height, width, num_filters]
  """
  outputs = layers.Conv2D(num_filters, 3, 2)(inputs)

  if batch_norm:
    outputs = layers.BatchNorm()(outputs)

  outputs = layers.LeakyReLU()(outputs)

  return outputs


def decoder_layer(inputs, num_filters, batch_norm=True, dropout=False):
  """Creates u-net decoder layer.

  If `inputs` is a list these tensors will be concatenated along their
  innermost dimension.

  Args:
    inputs: one or two input tensor of shape [batch, height, width, channels]
    num_filters: number of output channels
    batch_norm='True': use batch normalization after convolution
    dropout='False': use dropout after batch normalization
  Returns:
    outputs: output tensor of shape [batch, height, width, num_filters]
  """
  if type(inputs) is list:
    outputs = layers.Concatenate(axis=3)(inputs)
  else:
    outputs = inputs
  outputs = layers.Conv2DTranspose(num_filters, 3, 2)(outputs)
  if batch_norm:
    outputs = layers.BatchNorm()(outputs)
  if dropout:
    outputs = layers.Dropout()(outputs)

  return layers.LeakyReLU()(outputs)


def final_layer(inputs):
  """Creates u-net final layer.

  The original u-net architecture was built for binary classification,
  however in our case we want to have same output shape as the input,
  hence we replace the original final layer of u-net with a layer which
  decodes the previous output back to one channel.

  Args:
    inputs: input tensor of shape [batch, height, width, channels]
  Returns:
    outputs: output tensor of shape [batch, height, width, 1]
  """
  outputs = layers.Conv2DTranspose(1, 3, 1)(inputs)

  return layers.Activation(tf.nn.tanh)(outputs)


def generator_network(params):
  """Creates a u-net network.

  Args:
    inputs: input tensor of shape [batch, height, width, 1]
    num_filters: number of filters to act as base between layers
  Returns:
    outputs: output tensor of shape [batch, height, width, 1]
  """
  filters = [a * params.num_filters for a in [1, 2, 4, 8, 8]]

  inputs = outputs = layers.Input(shape=(None, None, 1))

  encoded = []
  for i, nf in enumerate(filters):
    x = inputs if i == 0 else encoded[-1]

    with tf.variable_scope(f'encode{i}'):
      encoded.append(encoder_layer(x, num_filters=nf, batch_norm=i != 0))

  decoded = []
  for i, nf in enumerate(reversed(filters)):
    x = encoded[-1] if i == 0 else [encoded[-1 - i], decoded[-1]]

    with tf.variable_scope(f'decode{i}'):
      decoded.append(decoder_layer(x, num_filters=nf, dropout=i in [0, 1]))

  with tf.variable_scope('finalize'):
    outputs = final_layer(decoded[-1])

  return layers.Network(inputs, outputs, name='unet')
