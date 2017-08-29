import os
import tensorflow as tf

from mrtoct import ioutil
from mrtoct import utils

def parse(example):
    image = ioutil.decode_example(example)
    return tf.cast(utils.normalize(image), tf.float32)

def filter_nans(ct, mr):
    return tf.logical_not(utils.has_nan(mr))

INCOMPLETE_SCALE = 1.3

def filter_incomplete(ct, mr):
    ct_sum = utils.count(tf.greater(ct, -1))
    mr_sum = utils.count(tf.greater(mr, -1))
    return tf.greater(tf.multiply(mr_sum, INCOMPLETE_SCALE), ct_sum)

def make_dataset(fileexpr):
    return tf.contrib.data.Dataset.list_files(fileexpr).flat_map(
        lambda filename: tf.contrib.data.TFRecordDataset(filename).map(parse))

def make_zipped_dataset(filepath):
    return tf.contrib.data.Dataset.zip((
        make_dataset(os.path.join(filepath, '*ct.tfrecord')),
        make_dataset(os.path.join(filepath, '*mr.tfrecord')),
    ))

class TrainValidIterator:

    def __init__(self, train_dataset, valid_dataset):
        handle = tf.placeholder(tf.string, shape=[])

        train_iterator = train_dataset.make_one_shot_iterator()
        valid_iterator = valid_dataset.make_one_shot_iterator()

        self._iterator = tf.contrib.data.Iterator.from_string_handle(handle,
            train_dataset.output_types, train_dataset.output_shapes)
        self._train_handle = train_iterator.string_handle()
        self._valid_handle = valid_iterator.string_handle()
        self._handle = handle

    def get_next(self):
        return self._iterator.get_next()

    def get_handle(self):
        return self._handle

    def get_train_handle(self):
        return self._train_handle

    def get_valid_handle(self):
        return self._valid_handle