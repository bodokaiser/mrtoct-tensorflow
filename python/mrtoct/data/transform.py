import tensorflow as tf

from mrtoct import ioutil


class Compose:
  """Composes list of transforms to single transform."""

  def __init__(self, transforms=[]):
    assert len(transforms) > 0
    self.transforms = transforms

  def __call__(self, *args):
    with tf.name_scope('compose'):
      for fn in self.transforms:
        if type(args) is tuple:
          args = fn(*args)
        else:
          args = fn(args)

      return args


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


class CenterPad:
  """Pads input symmetric to `shape`."""

  def __init__(self, shape):
    self.shape = shape

  def __call__(self, x):
    with tf.name_scope('center_pad'):
      target_shape = tf.convert_to_tensor(self.shape, name='target_shape')
      input_shape = tf.shape(x, name='input_shape')

      print(target_shape)
      print(input_shape)

      # TODO: check explicit if tensor shapes are compatible
      ndims = target_shape.shape.num_elements()

      paddings = []

      for i in range(ndims):
        offset = (target_shape[i] - input_shape[i]) / 2

        paddings.append(tf.stack([tf.floor(offset), tf.ceil(offset)]))

      return tf.pad(x, tf.to_int32(tf.stack(paddings)))


class CenterCrop:
  """Crops input symmetric to `shape`."""

  def __init__(self, shape):
    self.shape = shape

  def __call__(self, x):
    with tf.name_scope('center_crop'):
      off = tf.subtract(tf.shape(x), self.shape) // 2

      return tf.slice(x, off, self.shape)


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


class ExtractSlice:
  """Extracts a slice from `axis` at `index` from input."""

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, index, x):
    if self.axis == 0:
      return x[index]
    if self.axis == 1:
      return x[:, index]
    if self.axis == 2:
      return x[:, :, index]

    raise ValueError(f'axis should be 0, 1, 2 not {self.axis}')


class ExtractPatch:
  """Extracts a patch of `shape` centered at `index` from input."""

  def __init__(self, shape):
    self.shape = shape

  def __call__(self, index, x):
    with tf.name_scope('extract_patch'):
      index = tf.convert_to_tensor(index, name='index')
      shape = tf.convert_to_tensor(self.shape, name='shape')
      offset = tf.cast(tf.floor(shape / 2), index.dtype)

      start = []
      for i in range(index.get_shape().num_elements()):
        start.append(index[i] - offset[i])
      start.append(0)

      start = tf.stack(start, name='start')

      return tf.slice(x, start, shape)
