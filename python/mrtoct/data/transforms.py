import tensorflow as tf

from mrtoct import ioutil, util


class Compose:

  def __init__(self, transforms=[]):
    assert len(transforms) > 0
    self.transforms = transforms

  def __call__(self, x):
    with tf.name_scope('compose'):
      for fn in self.transforms:
        x = fn(x)

      return x


class CastType:

  def __init__(self, dtype=tf.float32):
    self.dtype = dtype

  def __call__(self, x):
    with tf.name_scope('cast_type'):
      return tf.cast(x, self.dtype)


class ExpandDim:

  def __init__(self, axis=-1):
    self.axis = axis

  def __call__(self, x):
    with tf.name_scope('expand_dim'):
      return tf.expand_dims(x, self.axis)


class CenterMean:

  def __call__(self, x):
    with tf.name_scope('center_mean'):
      x -= tf.reduce_min(x)
      x /= tf.reduce_max(x)

      return 2 * x - 1


class UncenterMean:

  def __call__(self, x):
    with tf.name_scope('uncenter_mean'):
      return (x + 1) / 2


class DecodeExample:

  def __init__(self, decoder=None):
    self.decoder = decoder if decoder is not None else ioutil.TFRecordDecoder()

  def __call__(self, x):
    with tf.name_scope('decode_example'):
      return self.decoder.decode(x)


class ExtractPatch:

  def __init__(self, shape):
    self.shape = shape

  def __call__(self, index, x):
    with tf.name_scope('extract_patch'):
      start = tf.subtract(index, tf.divide(index, 2))
      stop = tf.add(start, index)

      indices = util.meshgrid(start, stop, 1)
      indices_flat = tf.transpose(tf.reshape(indices, [-1]))

      return tf.gather_nd(x, indices_flat)
