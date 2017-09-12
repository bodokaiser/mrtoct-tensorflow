import tensorflow as tf


def _variable(name, shape, init):
    return tf.get_variable(name, shape, tf.float32, init, trainable=False)


class SparseMovingAverage:

    def __init__(self, shape, name):
        self._name = name
        self._shape = shape

    def apply(self, indices, values):
        with tf.variable_scope(self._name):
            self._value = _variable('value', self._shape,
                                    tf.zeros_initializer())
            self._weight = _variable('weight', self._shape,
                                     tf.ones_initializer())
            self._average = tf.div(self._value, self._weight)

            batch_size = tf.shape(indices)[0]

            indices = tf.reshape(indices, [-1, 5])
            value_updates = tf.reshape(values, [-1])
            weight_updates = tf.cast(tf.greater(value_updates, 0), tf.float32)

            return tf.group(
                tf.scatter_nd_add(self._value, indices, value_updates),
                tf.scatter_nd_add(self._weight, indices, weight_updates))

    def average(self):
        return self._average
