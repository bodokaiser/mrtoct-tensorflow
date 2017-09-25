import tensorflow as tf


def meshgrid_2d(start, stop, delta=1):
  with tf.name_scope('meshgrid_2d'):
    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop)

    shape1 = start.shape.as_list()
    shape2 = stop.shape.as_list()

    if shape1 != shape2 or len(shape1) != 1 or shape1[0] != 2:
      raise ValueError(f'start and stop should have shape [2] not {shape1}')

    cols = tf.range(start[-1], stop[-1], delta, dtype=tf.int32)
    rows = tf.range(start[-2], stop[-2], delta, dtype=tf.int32)

    return tf.stack(tf.meshgrid(cols, rows, indexing='ij'), 2)


def meshgrid_3d(start, stop, delta=1):
  with tf.name_scope('meshgrid_3d'):
    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop)

    shape1 = start.shape.as_list()
    shape2 = stop.shape.as_list()

    if shape1 != shape2 or len(shape1) != 1 or shape1[0] != 3:
      raise ValueError(f'start and stop should have shape [3] not {shape1}')

    cols = tf.range(start[-1], stop[-1], delta, dtype=tf.int32)
    rows = tf.range(start[-2], stop[-2], delta, dtype=tf.int32)
    slices = tf.range(start[-3], stop[-3], delta, dtype=tf.int32)

    return tf.stack(tf.meshgrid(slices, cols, rows, indexing='ij'), 3)


def spatial_gradient_3d(volume):
  with tf.name_scope('spatial_gradient_3d'):
    volume = tf.convert_to_tensor(volume)

    if volume.get_shape().ndims != 5:
      raise ValueError(f'volume should have rank 5 not {ndims}')

    return [volume[:, 1:, :, :, :] - volume[:, :-1, :, :, :],
            volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :],
            volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]]
