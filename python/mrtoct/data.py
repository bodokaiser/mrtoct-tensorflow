import os
import tensorflow as tf

from mrtoct import ioutil
from mrtoct.utils import count, has_nan, process, normalize

DEFAULT_DECODER = ioutil.TFRecordDecoder()

def parse(example):
    volume = DEFAULT_DECODER.decode(example)
    volume = tf.image.convert_image_dtype(volume, tf.float32)
    return process(normalize(volume))

def filter_nans(mr, ct):
    return tf.logical_not(tf.logical_or(has_nan(ct), has_nan(mr)))

INCOMPLETE_SCALE = 1.4

def filter_incomplete(mr, ct):
    mr_sum = count(tf.greater(mr, tf.reduce_min(mr)))
    ct_sum = count(tf.greater(ct, tf.reduce_min(ct)))
    return tf.greater(tf.multiply(mr_sum, INCOMPLETE_SCALE), ct_sum)

def make_dataset(fileexpr):
    ctype = tf.python_io.TFRecordOptions.get_compression_type_string(
        ioutil.TFRecordOptions)

    return tf.contrib.data.Dataset.list_files(fileexpr).flat_map(
        lambda fname: tf.contrib.data.TFRecordDataset(fname, ctype).map(parse))

def make_zipped_dataset(filepath):
    return tf.contrib.data.Dataset.zip((
        make_dataset(os.path.join(filepath, 're-co-mr-*.tfrecord')),
        make_dataset(os.path.join(filepath, 're-ct-*.tfrecord')),
    ))

def make_iterator_from_handle(handle, dataset):
    return tf.contrib.data.Iterator.from_string_handle(handle,
        dataset.output_types, dataset.output_shapes)