import tensorflow as tf


def meshgrid(start, stop, delta=1):
  with tf.name_scope('meshgrid'):
    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop)

    shape1 = start.shape.as_list()
    shape2 = stop.shape.as_list()

    if len(shape1) != 1 or shape1 != shape2:
      raise ValueError('start and stop shape should have same rank 1 shape')

    ndim = shape1[0]

    rows = tf.range(start[-2], stop[-2], delta, dtype=tf.int32)
    cols = tf.range(start[-1], stop[-1], delta, dtype=tf.int32)

    if ndim == 2:
      return tf.stack(tf.meshgrid(cols, rows, indexing='ij'), 2)

    slices = tf.range(start[-3], stop[-3], delta, dtype=tf.int32)

    if ndim == 3:
      return tf.stack(tf.meshgrid(slices, cols, rows, indexing='ij'), 3)

    raise ValueError('start and stop shape dimension need to be 2 or 3')


def gradient(x):
  ndims = x.get_shape().ndims
  diffs = []

  if ndims == 4:
    diffs.append(tf.abs(x[:, 1:, :, :] - x[:, :-1, :, :]))
    diffs.append(tf.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
  if ndims == 5:
    diffs.append(tf.abs(x[:, 1:, :, :, :] - x[:, :-1, :, :, :]))
    diffs.append(tf.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    diffs.append(tf.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))

  return diffs
