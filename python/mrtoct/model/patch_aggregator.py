import tensorflow as tf


class PatchAggregator3D:

  def __init__(self, shape, name):
    self.name = name
    self.shape = shape

  def update(self, indices, values):
    shape = self.shape

    if indices.get_shape().ndims != 4:
      raise ValueError('indices must be of rank 4')
    if values.get_shape().ndims != 4:
      raise ValueError('values must be of rank 4')

    with tf.variable_scope(f'patch_aggregator{self.name}'):
      batch = tf.shape(indices)[0]

      self._value = tf.get_variable('value', shape, tf.float32,
                                    trainable=False,
                                    initializer=tf.zeros_initializer())
      self._weight = tf.get_variable('weight', shape, tf.float32,
                                     trainable=False,
                                     initializer=tf.ones_initializer())
      self._average = self._value / self._weight

      indices = tf.reshape(indices, [batch, 3, -1])
      value_updates = tf.reshape(values, [batch, -1])
      weight_updates = tf.cast(tf.greater(value_updates, 0), tf.float32)

      return tf.group(
          tf.scatter_nd_add(self._value, indices, value_updates),
          tf.scatter_nd_add(self._weight, indices, weight_updates))

  def average(self):
    return self._average
