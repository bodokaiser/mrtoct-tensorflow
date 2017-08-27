import tensorflow as tf

from mrtoct import ioutil

def _parse(example):
    image = ioutil.decode_example(example)

    image -= tf.reduce_min(image)
    image /= tf.reduce_max(image)
    image = 2*image - 1

    return tf.cast(image, tf.float32)

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


def make_dataset(fileexpr):
    return tf.contrib.data.Dataset.list_files(fileexpr).flat_map(
        lambda filename: tf.contrib.data.TFRecordDataset(filename).map(_parse))