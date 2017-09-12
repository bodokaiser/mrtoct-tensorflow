import functools
import operator
import tensorflow as tf

from mrtoct import ioutil
from mrtoct import util


DEFAULT_CTYPE = 'GZIP'


def filename_to_tfrecord(ctype=None):
    if ctype is None:
        ctype = DEFAULT_CTYPE

    def map_filename_to_tfrecord(filename):
        return tf.contrib.data.TFRecordDataset(filename, ctype)

    return map_filename_to_tfrecord


def tfrecord_to_tensor(decoder=None):
    if decoder is None:
        decoder = ioutil.TFRecordDecoder()

    def map_tfrecord_to_tensor(tfrecord):
        return decoder.decode(tfrecord)

    return map_tfrecord_to_tensor


def extract_patch(pshape):
    def map_extract_patch(idvolume, center):
        id, volume = idvolume

        start = tf.subtract(center, tf.div(pshape[:-1], 2))
        stop = tf.add(start, pshape[:-1])

        indices = util.meshgrid(start, stop, 1, 3)
        patch = tf.reshape(tf.gather_nd(volume, indices), pshape)

        num_indices = functools.reduce(operator.mul, pshape)

        id_indices = id * tf.ones([num_indices, 1], tf.int64)
        ch_indices = tf.zeros([num_indices, 1], tf.int64)

        new_indices = tf.concat([
            tf.cast(id_indices, tf.int32),
            indices,
            tf.cast(ch_indices, tf.int32)], 1)

        return new_indices, patch

    return map_extract_patch
