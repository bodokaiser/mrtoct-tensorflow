import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def encode(image):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image.astype(np.int32).tobytes()),
        'shape': _int64_feature(image.shape),
    })).SerializeToString()

def decode(example):
    features = tf.parse_single_example(example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([2], tf.int64),
    })

    shape = tf.cast(features['shape'], tf.int32)
    image = tf.reshape(tf.decode_raw(features['image'], tf.int32), shape)

    return tf.expand_dims(image, -1)