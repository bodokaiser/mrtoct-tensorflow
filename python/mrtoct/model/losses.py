import tensorflow as tf


EPSILON = 1e-12


def mae(targets, outputs):
    with tf.name_scope('mae'):
        return tf.reduce_mean(tf.abs(targets - outputs))


def mse(targets, outputs):
    with tf.name_scope('mse'):
        return tf.reduce_mean(tf.square(targets - outputs))


def adv_d(fake_score, real_score):
    with tf.name_scope('adv'):
        return tf.reduce_mean(
            - tf.log(real_score + EPSILON)
            - tf.log(1 - fake_score + EPSILON))


def adv_g(fake_score):
    with tf.name_scope('adv'):
        return tf.reduce_mean(-tf.log(fake_score + EPSILON))
