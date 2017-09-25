import tensorflow as tf


class PatchAggregator:

  def __init__(self, shape, name):
    self.name = name
    self.shape = shape

  def update(self, indices, values):
    with tf.name_scope('patch_aggregator'):
      with tf.variable_scope(self.name):
        self._value = tf.get_variable('value', self.shape, tf.float32,
                                      tf.zeros_initializer(), trainable=False)
        self._weight = tf.get_variable('weight', self.shape, tf.float32,
                                       tf.ones_initializer(), trainable=False)
        self._average = self._value / self._weight

        indices = tf.transpose(tf.reshape(indices, [3, -1]))
        value_updates = tf.reshape(values, [-1])
        weight_updates = tf.cast(tf.greater(value_updates, 0), tf.float32)

        return tf.group(
            tf.scatter_nd_add(self._value, indices, value_updates),
            tf.scatter_nd_add(self._weight, indices, weight_updates))

  def average(self):
    return self._average
