import os
import tensorflow as tf

from mrtoct.ioutil import tfrecord
from mrtoct.utils import count, has_nan

def parse(example):
    image = tfrecord.decode(example)
    return tf.image.per_image_standardization(image)

def filter_nans(mr, ct):
    return tf.logical_not(tf.logical_or(has_nan(ct), has_nan(mr)))

INCOMPLETE_SCALE = 1.3

def filter_incomplete(mr, ct):
    ct_sum = count(tf.greater(ct, -1))
    mr_sum = count(tf.greater(mr, -1))
    return tf.greater(tf.multiply(ct_sum, INCOMPLETE_SCALE), mr_sum)

def make_dataset(fileexpr):
    return tf.contrib.data.Dataset.list_files(fileexpr).flat_map(
        lambda filename: tf.contrib.data.TFRecordDataset(filename).map(parse))

def make_zipped_dataset(filepath):
    return tf.contrib.data.Dataset.zip((
        make_dataset(os.path.join(filepath, '*mr.tfrecord')),
        make_dataset(os.path.join(filepath, '*ct.tfrecord')),
    ))

def make_iterator_from_handle(handle, dataset):
    return tf.contrib.data.Iterator.from_string_handle(handle,
        dataset.output_types, dataset.output_shapes)