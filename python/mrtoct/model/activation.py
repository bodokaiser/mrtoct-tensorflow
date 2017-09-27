import tensorflow as tf


def lrelu(x, leakiness=.02):
  with tf.name_scope('leaky_relu'):
    return tf.where(tf.less(x, .0), leakiness * x, x)
