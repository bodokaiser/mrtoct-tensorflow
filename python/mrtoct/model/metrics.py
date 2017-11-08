import tensorflow as tf


def peak_signal_to_noise_ratio(targets, outputs):
  with tf.name_scope('peak_signal_to_noise_ratio'):
    ssum = tf.reduce_sum(tf.square(targets - outputs), 0)
    ndim = tf.reduce_prod(targets.shape[1:])
    mse = ssum / tf.to_float(ndim)

    return tf.reduce_mean(- 10 * tf.log(mse))


def structural_similarity_index(targets, outputs):
  pass
