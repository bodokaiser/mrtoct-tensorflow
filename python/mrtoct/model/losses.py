import tensorflow as tf

from mrtoct import util


EPSILON = 1e-12


def mae(targets, outputs):
  with tf.name_scope('mae'):
    return tf.reduce_mean(tf.abs(targets - outputs))


def mse(targets, outputs):
  with tf.name_scope('mse'):
    return tf.reduce_mean(tf.square(targets - outputs))


def adv_d(fake_score, real_score):
  with tf.name_scope('adv'):
    r_log = -tf.log(real_score + EPSILON)
    f_log = -tf.log(1 - fake_score + EPSILON)

    return tf.reduce_mean(r_log + f_log)


def adv_g(fake_score):
  with tf.name_scope('adv'):
    return tf.reduce_mean(-tf.log(fake_score + EPSILON))


def gdl(targets, outputs):
  with tf.name_scope('gdl'):
    targets_grad = util.spatial_gradient_3d(targets)
    outputs_grad = util.spatial_gradient_3d(outputs)

    loss = tf.zeros([])

    for i in range(3):
      loss += mse(tf.abs(targets_grad[i]), tf.abs(outputs_grad[i]))

    return loss
