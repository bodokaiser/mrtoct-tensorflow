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


class Lambda:
  """Executes a lambda function."""

  def __init__(self, func):
    self.func = func

  def __call__(self, *args):
    return self.func(*args)


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
      x = tf.to_float(x)
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


class CropOrPad2D:
  """Crops or pads image."""

  def __init__(self, height, width):
    self.height = height
    self.width = width

  def __call__(self, x):
    with tf.name_scope('crop_or_pad_2d'):
      x = tf.image.resize_image_with_crop_or_pad(x, self.height, self.width)
      x = tf.reshape(x, [self.height, self.width])

      return x


class CenterPad3D:
  """Resizes volume by pad."""

  def __init__(self, depth, height, width):
    self.height = height
    self.width = width
    self.depth = depth

  def __call__(self, x):
    with tf.name_scope('center_pad_3d'):
      input_shape = tf.shape(x, name='input_shape')
      target_shape = tf.constant([
          self.depth, self.height, self.width],
          name='target_shape')

      padding = []

      for i in range(3):
        offset = (target_shape[i] - input_shape[i]) / 2
        padding.append(tf.stack([
            tf.floor(offset), tf.ceil(offset)]))

      padding.append([0, 0])

      return tf.pad(x, tf.to_int32(tf.stack(padding)))


class IndexCrop3D:
  """Crops patch of `shape` centered at `index` from input."""

  def __init__(self, shape, index):
    self.index = index
    self.shape = shape

  def __call__(self, x):
    with tf.name_scope('index_crop_3d'):
      index = tf.convert_to_tensor(self.index, name='index')
      shape = tf.convert_to_tensor(self.shape, name='shape')
      offset = tf.cast(tf.floor(shape / 2), index.dtype)

      start = []
      for i in range(3):
        start.append(index[i] - offset[i])
      start.append(0)

      start = tf.stack(start, name='start')
      stop = tf.concat([shape, [-1]], axis=0, name='stop')

      return tf.slice(x, start, stop)
