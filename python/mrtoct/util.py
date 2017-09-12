import tensorflow as tf


def meshgrid(start, stop, delta, n):
    ranges = [tf.range(start[i], stop[i], delta, tf.int32) for i in range(n)]

    return tf.transpose(tf.reshape(tf.meshgrid(*ranges), [n, -1]))
