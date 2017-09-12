import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class Encoder:

    def encode(self, volume):
        buf = volume.astype(np.int16).tobytes()

        with tf.name_scope('encode'):
            return tf.train.Example(features=tf.train.Features(feature={
                'volume/encoded': _bytes_feature(buf),
                'volume/height': _int64_feature(volume.shape[0]),
                'volume/width': _int64_feature(volume.shape[1]),
                'volume/depth': _int64_feature(volume.shape[2]),
            })).SerializeToString()


class Decoder:

    def __init__(self):
        self.features = {
            'volume/encoded': tf.FixedLenFeature((), tf.string),
            'volume/height': tf.FixedLenFeature((), tf.int64),
            'volume/width': tf.FixedLenFeature((), tf.int64),
            'volume/depth': tf.FixedLenFeature((), tf.int64),
        }

    def decode(self, example):
        with tf.name_scope('decode'):
            features = tf.parse_single_example(example, features=self.features)

            return tf.reshape(tf.decode_raw(
                features['volume/encoded'], tf.int16), [
                tf.cast(features['volume/height'], tf.int32),
                tf.cast(features['volume/width'], tf.int32),
                tf.cast(features['volume/depth'], tf.int32),
            ])
