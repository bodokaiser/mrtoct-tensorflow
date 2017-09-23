import tensorflow as tf


def zero_center_mean(x):
    return tf.subtract(tf.multiply(x, 2), 1)


def uncenter_zero_mean(x):
    return tf.divide(tf.add(x, 1), 2)


def tensor_value_range():
    def normalize_value_range(tensor):
        tensor = tf.cast(tensor, tf.float32)
        tensor -= tf.reduce_min(tensor)
        tensor /= tf.reduce_max(tensor)

        return tensor

    return normalize_value_range


def tensor_shape(shape):
    def normalize_shape(tensor):
        off = tf.divide(tf.subtract(shape, tf.shape(tensor)), 2)
        pad = [tf.stack([tf.floor(off[i]), tf.ceil(off[i])]) for i in range(3)]

        tensor = tf.pad(tensor, tf.cast(tf.stack(pad), tf.int32))
        tensor = tf.reshape(tensor, shape)

        return tensor

    return normalize_shape
