import tensorflow as tf

def has_nan(tensor):
    """Returns True if tensor has a NaN value else False."""
    with tf.name_scope('has_nan'):
        return tf.greater(tf.reduce_sum(tf.cast(tf.is_nan(tensor), tf.float32)), 0)

def count(condition):
    """Returns the number of elements for which condition is True."""
    with tf.name_scope('count'):
        return tf.reduce_sum(tf.cast(condition, tf.float32))

def process(images):
    """Converts images range from [0,1] to [-1,+1]."""
    with tf.name_scope('process'):
        return 2 * images - 1

def deprocess(images):
    """Converts images range from [-1,+1] to [0,1]."""
    with tf.name_scope('deprocess'):
        return (images + 1) / 2

def normalize(images):
    """Normalizes images range to [0,1]."""
    with tf.name_scope('normalize'):
        images -= tf.reduce_min(images)
        images /= tf.reduce_max(images)
        return images