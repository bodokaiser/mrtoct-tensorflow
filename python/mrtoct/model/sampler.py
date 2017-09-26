import tensorflow as tf

from mrtoct import util


def sample_meshgrid_3d(start, stop, delta):
  """Samples indices from 3d meshgrid.

  Args:
    start: tensor or array to start to sample
    end: tensor or array to stop to sample
    delta: tensor or int as sample distance
  Returns:
    dataset
  """
  with tf.name_scope('sample_meshgrid'):
    indices = util.meshgrid(start, stop, delta)

    return tf.contrib.data.Dataset.from_tensor_slices(indices)


def sample_uniform_3d(start, stop, size):
  """Samples indices from 3d uniform distribution.

  Args:
    start: tensor or array to start to sample
    end: tensor or array to stop to sample
    size: number of samples
  Returns:
    dataset
  """
  with tf.name_scope('sample_uniform'):
    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop)

    if start.get_shape().ndims != stop.get_shape().ndims != 3:
      raise ValueError('start and stop should have shape [3]')

    indices = tf.stack([
        tf.random_uniform([size], start[0], stop[0], tf.int32),
        tf.random_uniform([size], start[1], stop[1], tf.int32),
        tf.random_uniform([size], start[2], stop[2], tf.int32),
    ], 1)

    return tf.contrib.data.Dataset.from_tensor_slices(indices)
