import tensorflow as tf

from mrtoct import util


def sample_meshgrid(start, stop, delta):
  with tf.name_scope('sample_meshgrid'):
    indices = util.meshgrid(start, stop, delta)

    return tf.contrib.data.Dataset.from_tensor_slices(indices)


def sample_uniform(start, stop, size):
  with tf.name_scope('sample_uniform'):
    with tf.control_dependencies([tf.assert_rank(tf.rank(start), 3),
                                  tf.assert_rank(tf.rank(stop), 3)]):
      indices = tf.stack([
          tf.random_uniform([size], start[0], stop[0], tf.int32),
          tf.random_uniform([size], start[1], stop[1], tf.int32),
          tf.random_uniform([size], start[2], stop[2], tf.int32),
      ], 1)

      return tf.contrib.data.Dataset.from_tensor_slices(indices)
