import tensorflow as tf

from mrtoct import util

mse = tf.losses.mean_squared_error


def gradient_difference_loss_2d(targets, outputs):
  with tf.name_scope('gradient_difference_loss_2d'):
    return tf.reduce_sum(tf.image.total_variation(targets - outputs))


def gradient_difference_loss_3d(targets, outputs):
  with tf.name_scope('gradient_difference_loss_3d'):
    grad1 = util.spatial_gradient_3d(targets)
    grad2 = util.spatial_gradient_3d(outputs)

    return mse(tf.abs(grad1[0]), tf.abs(grad2[0]), loss_collection=None) + \
        mse(tf.abs(grad1[1]), tf.abs(grad2[1]), loss_collection=None) + \
        mse(tf.abs(grad1[2]), tf.abs(grad2[2]), loss_collection=None)
