import tensorflow as tf

def has_nan(tensor):
    """Returns True if tensor has a NaN value else False."""
    return tf.greater(tf.reduce_sum(tf.cast(tf.is_nan(tensor), tf.float32)), 0)

def count(condition):
    """Returns the number of elements for which condition is True."""
    return tf.reduce_sum(tf.cast(condition, tf.float32))

def normalize(tensor):
    """Normalizes the tensor to the range [-1, +1]."""
    tensor -= tf.reduce_min(tensor)
    tensor /= tf.reduce_max(tensor)
    return 2*tensor - 1