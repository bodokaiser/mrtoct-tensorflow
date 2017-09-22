import tensorflow as tf


def meshgrid(start, stop, delta, n):
    ranges = [tf.range(start[i], stop[i], delta, tf.int32) for i in range(n)]

    return tf.transpose(tf.reshape(tf.meshgrid(*ranges), [n, -1]))


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
