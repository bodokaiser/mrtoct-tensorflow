import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def encode_example(mrslice, ctslice):
    return tf.train.Example(features=tf.train.Features(feature={
        'ct': _bytes_feature(ctslice.tobytes()),
        'mr': _bytes_feature(mrslice.tobytes()),
    })).SerializeToString()

def decode_example(example):
    features = tf.parse_single_example(example, features={
        'ct': tf.FixedLenFeature([], tf.string),
        'mr': tf.FixedLenFeature([], tf.string),
    })
    shape = [512, 512, -1]

    ctslice = tf.reshape(tf.decode_raw(features['ct'], tf.uint16), shape)
    mrslice = tf.reshape(tf.decode_raw(features['mr'], tf.uint16), shape)

    return ctslice, mrslice
