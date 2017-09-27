import tensorflow as tf


def lrelu(x, leakiness=.02):
  with tf.name_scope('leaky_relu'):
    return tf.where(tf.less(x, .0), leakiness * x, x)


class UNet:

  def __init__(self, inputs):
    self._inputs = inputs
    self._encoders = []
    self._decoders = []
    self._final = None

  def add_encoder(self, num_filters, batch_norm=True):
    x = self._encoders[-1] if len(self._encoders) > 0 else self._inputs

    with tf.name_scope(f'encoder{len(self._encoders)}'):
      x = tf.layers.conv2d(x, num_filters, 3, 2, 'SAME')

      if batch_norm:
        x = tf.layers.batch_normalization(x)
      x = lrelu(x)

    self._encoders.append(x)

    return x

  def add_decoder(self, num_filters, batch_norm=True, dropout=False):
    enc = self._encoders[-len(self._decoders) - 1]
    dec = self._decoders[-1] if len(self._decoders) > 0 else None

    with tf.name_scope(f'decoder{len(self._decoders)}'):
      x = tf.concat([dec, enc], 3) if dec is not None else enc
      x = tf.layers.conv2d_transpose(
          self.get_layer(), num_filters, 3, 2, 'SAME')

      if batch_norm:
        x = tf.layers.batch_normalization(x)
      if dropout:
        x = tf.layers.dropout(x)
      x = lrelu(x)

    self._decoders.append(x)

    return x

  def finalize(self):
    x = self._decoders[-1]

    with tf.name_scope('final'):
      x = tf.layers.conv2d_transpose(x, 1, 3, 1, 'SAME')
      x = tf.nn.tanh(x)

    return x
