import tensorflow as tf

def model(x):
    """Minimal trainable model to test training setup."""

    return tf.layers.conv2d(x, 1, 3, 1, 'SAME')