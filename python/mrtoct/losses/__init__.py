import tensorflow as tf

def l2(outputs, targets):
    return tf.square(1+tf.abs(tf.subtract(outputs, targets)))
