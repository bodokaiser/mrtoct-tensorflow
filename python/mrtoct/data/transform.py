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
    def map_extract_patch(volume, center):
        start = tf.subtract(center, tf.div(pshape, 2))

        return tf.slice(volume, start, pshape)

    return map_extract_patch
