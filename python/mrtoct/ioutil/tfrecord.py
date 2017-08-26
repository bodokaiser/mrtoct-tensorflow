import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def encode_example(mrslice, ctslice):
    assert len(mrslice.shape) == 2
    assert len(ctslice.shape) == 2

    return tf.train.Example(features=tf.train.Features(feature={
        'ct/image': _bytes_feature(ctslice.astype(np.int32).tobytes()),
        'ct/shape': _int64_feature(ctslice.shape),

        'mr/image': _bytes_feature(mrslice.astype(np.int32).tobytes()),
        'mr/shape': _int64_feature(mrslice.shape),
    })).SerializeToString()

def decode_example(example):
    features = tf.parse_single_example(example, features={
        'ct/image': tf.FixedLenFeature([], tf.string),
        'ct/shape': tf.FixedLenFeature([2], tf.int64),

        'mr/image': tf.FixedLenFeature([], tf.string),
        'mr/shape': tf.FixedLenFeature([2], tf.int64),
    })

    ctslice = tf.reshape(tf.decode_raw(features['ct/image'], tf.int32),
        tf.cast(features['ct/shape'], tf.int32))

    mrslice = tf.reshape(tf.decode_raw(features['mr/image'], tf.int32),
        tf.cast(features['mr/shape'], tf.int32))

    return ctslice, mrslice
