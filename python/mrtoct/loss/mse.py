import tensorflow as tf

slim = tf.contrib.slim

def mse(outputs, targets):
    return slim.losses.sum_of_squares(outputs, targets).get_total_loss()
