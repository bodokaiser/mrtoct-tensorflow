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

def index_to_indices(index):
    rows = tf.range(index[0]-16, index[0]+16, dtype=tf.int32)
    cols = tf.range(index[1]-16, index[1]+16, dtype=tf.int32)
    slices = tf.range(index[2]-16, index[2]+16, dtype=tf.int32)

    k, i, j = tf.meshgrid(slices, cols, rows, indexing='ij')

    return tf.stack([tf.reshape(i, [-1]), tf.reshape(j, [-1]), tf.reshape(k, [-1])])

def extract_patch(pshape):
    def map_extract_patch(volume, center):
        #start = tf.subtract(center, tf.div(pshape, 2))

        indices = tf.transpose(tf.reshape(index_to_indices(center), [3,-1]))

        return tf.reshape(tf.gather_nd(volume, indices), pshape)
        #return tf.slice(volume, start, pshape)

    return map_extract_patch
