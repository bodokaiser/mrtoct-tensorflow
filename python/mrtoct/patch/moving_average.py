import tensorflow as tf


class SparseMovingAverage:
  """Calculates moving average of sparse updates into target volume.

  The 'difficulty' in patch aggregation is that you need to do some moving
  average calculations as patches could overlap. In this case we keep track
  of two variables with target shape: The first one is the cummulation of the
  patches the second the cummulation of the patch masks. The second one can
  be used as weight to find the average between to overlapping patches as
  some patches may have more overlap than others.
  """

  def __init__(self, shape, name='', dtype=tf.float32):
    """Creates a new SparseMovingAverage.

    Args:
      shape: target shape to do sparse updates to
      name: used for variable and name prefixes
    """
    self._name = name
    self._shape = shape
    self._dtype = dtype
    self._built = False
    self._value = None
    self._weight = None
    self._average = None

  def _build(self):
    """Adds variables to computation graph."""
    self._value = tf.get_variable(f'{self._name}/value',
                                  shape=self._shape,
                                  dtype=self._dtype,
                                  trainable=False,
                                  initializer=tf.zeros_initializer())
    self._weight = tf.get_variable(f'{self._name}/weight',
                                   shape=self._shape,
                                   dtype=self._dtype,
                                   trainable=False,
                                   initializer=tf.zeros_initializer())
    self._built = True

  def update(self, indices, values):
    """Returns op to do sparse update.

    To elaborate on this one, lets consider two examples of 5x5x5 patches
    with target volume 100x100x100:

    1. We have 10 patches batched together into [10, 5, 5, 5] then our indices
    need to be [10, 5, 5, 5, 3].
    2. We have a single patch of shape [5, 5, 5] and indices [5, 5, 5, 3].

    As a third example consider a flattened patch:

    3. We have 4 5x5x5 patches flattened into [500] then we need indices of
    shape [500, 3].

    In other words for the innermost dimension we need to provide an index
    into the target shape for every value but the outer dimensions are
    arbitrary as long as they are the same for indices and values.

    Args:
      indices
      values
    Returns:
      op
    """""
    if not self._built:
      self._build()

    with tf.name_scope(f'{self._name}/patch_aggregator'):
      weights = tf.to_float(tf.greater(values, 0))

      if self._average is None:
        with tf.name_scope('average'):
          cond = tf.not_equal(self._weight, 0)
          ones = tf.ones_like(self._weight)
          weight = tf.where(cond, self._weight, ones)
          self._average = self._value / weight
      return tf.group(
          tf.scatter_nd_add(self._value, indices, values),
          tf.scatter_nd_add(self._weight, indices, weights))

  def average(self):
    """Returns tensor with volume."""
    if self._average is None:
      raise RuntimeError('you need to call ".update" at least once')

    return self._average

  def initializer(self):
    """Returns initializer op for variables."""
    if not self._built:
      raise RuntimeWarning('you need to call ".update" at least once')

    return tf.variables_initializer([self._value, self._weight])
