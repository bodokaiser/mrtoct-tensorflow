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


def pix2pix_generator(x, params):
  unet = UNet(x)

  for i, a in enumerate([1, 2, 4, 8, 8]):
    unet.add_encoder(num_filters=a * params.num_filters, batch_norm=i == 0)
  for i, a in enumerate([8, 8, 4, 2, 1]):
    unet.add_decoder(num_filters=a * params.num_filters, dropout=i in [0, 1])

  return unet.finalize()


def pix2pix(x1, x2, params):
  nf = params.num_filters

  x = tf.concat([x1, x2], 3)

  for i, s in enumerate([nf, 2 * nf, 4 * nf]):
    x = tf.layers.conv2d(x, s, 4, 2, 'SAME', name=f'conv{i}',
                         activation=lrelu)

  x = tf.layers.conv2d(x, 8 * nf, 4, 2, 'VALID',
                       name='conv3', activation=lrelu)
  x = tf.layers.conv2d(x, 1, 4, 1, 'VALID',
                       name='conv4', activation=lrelu)

  return tf.nn.sigmoid(x)


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
