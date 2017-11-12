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
  """Resizes image by crop or pad."""

  def __init__(self, height, width):
    self.height = height
    self.width = width

  def __call__(self, x):
    with tf.name_scope('crop_or_pad_2d'):
      x = tf.image.resize_image_with_crop_or_pad(x, self.height, self.width)
      x = tf.reshape(x, [self.height, self.width])

      return x


class RandomRotate:
  """Rotates image pair batch by random angle."""

  def __call__(self, x, y):
    with tf.name_scope('random_rotate'):
      angles = tf.random_uniform([1], 0, 360)
      images = tf.contrib.image.rotate(tf.stack([x, y]), angles[0])

      return images[0], images[1]


class ExtractPatch:
  """Extracts a patch of `shape` centered at `index` from input."""

  def __init__(self, shape, index):
    self.index = index
    self.shape = shape

  def __call__(self, x):
    with tf.name_scope('extract_patch'):
      index = tf.convert_to_tensor(self.index, name='index')
      shape = tf.convert_to_tensor(self.shape, name='shape')
      offset = tf.cast(tf.floor(shape / 2), index.dtype)

      start = []
      for i in range(index.get_shape().num_elements()):
        start.append(index[i] - offset[i])
      start.append(0)

      start = tf.stack(start, name='start')

      return tf.slice(x, start, shape)
