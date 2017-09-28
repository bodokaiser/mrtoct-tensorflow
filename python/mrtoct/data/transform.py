import tensorflow as tf

from mrtoct import ioutil


class Compose:
  """Composes list of transforms to single transform."""

  def __init__(self, transforms=[]):
    assert len(transforms) > 0
    self.transforms = transforms

  def __call__(self, x):
    with tf.name_scope('compose'):
      for fn in self.transforms:
        x = fn(x)

      return x


class CastType:
  """Casts input to given type."""

  def __init__(self, dtype=tf.float32):
    self.dtype = dtype

  def __call__(self, x):
    with tf.name_scope('cast_type'):
      return tf.cast(x, self.dtype)


class ExpandDims:
  """Expands input at given axis."""

  def __init__(self, axis=-1):
    self.axis = axis

  def __call__(self, x):
    with tf.name_scope('expand_dim'):
      return tf.expand_dims(x, self.axis)


class Normalize:
  """Normalizes input to [0,1]."""

  def __call__(self, x):
    with tf.name_scope('normalize'):
      x -= tf.reduce_min(x)
      x /= tf.reduce_max(x)

      return x


class CenterMean:
  """Normalizes input to [-1,1]."""

  def __call__(self, x):
    with tf.name_scope('center_mean'):
      return 2 * x - 1


class UncenterMean:
  """Unnormalizes input to [0,1]."""

  def __call__(self, x):
    with tf.name_scope('uncenter_mean'):
      return (x + 1) / 2


class DecodeExample:
  """Decodes a tfrecord string with `decoder`."""

  def __init__(self, decoder=None):
    self.decoder = decoder if decoder is not None else ioutil.TFRecordDecoder()

  def __call__(self, x):
    with tf.name_scope('decode_example'):
      return self.decoder.decode(x)


class ExtractPatch3D:
  """Extracts a 3d patch of `shape` centered at `index` from input."""

  def __init__(self, shape):
    self.shape = shape

  def __call__(self, index, x):
    with tf.name_scope('extract_patch'):
      index = tf.convert_to_tensor(index)
      shape = tf.convert_to_tensor(self.shape)
      start = index - tf.cast(tf.floor(shape / 2), index.dtype)

      return tf.slice(x, start, shape)
