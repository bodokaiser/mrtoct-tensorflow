import tensorflow as tf

def leaky_relu(x, leakiness=.02):
    return tf.where(tf.less(x, .0), leakiness * x, x, name='leaky_relu')