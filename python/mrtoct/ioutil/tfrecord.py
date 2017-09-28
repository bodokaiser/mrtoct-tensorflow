import numpy as np
import tensorflow as tf


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


Options = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.GZIP)


class Encoder:
  """Encodes a numpy array as tfrecord."""

  def encode(self, volume):
    """Encodes a numpy array as serialized tfrecord string.

    Args:
      volume: numpy array
    Returns:
      tfrecord: tfrecord serialized to string
    """
    with tf.name_scope('encode'):
      return tf.train.Example(features=tf.train.Features(feature={
          'volume/encoded': _bytes_feature([
              volume.astype(np.int32).tobytes()]),
          'volume/shape': _int64_feature(volume.shape),
          'volume/vmin': _int64_feature([volume.min()]),
          'volume/vmax': _int64_feature([volume.max()]),
      })).SerializeToString()


class Decoder:
  """Decodes a tfrecord as numpy array."""

  def __init__(self):
    self.features = {
        'volume/encoded': tf.FixedLenFeature((), tf.string),
        'volume/shape': tf.VarLenFeature(tf.int64),
        'volume/vmin': tf.FixedLenFeature((), tf.int64),
        'volume/vmax': tf.FixedLenFeature((), tf.int64),
    }

  def decode(self, example):
    """Decodes a tfrecord string to numpy array.

    Args:
      example: tfrecord serialized to string
    Returns:
      volume: numpy array
    """
    with tf.name_scope('decode'):
      features = tf.parse_single_example(example, features=self.features)

      return tf.reshape(tf.decode_raw(features['volume/encoded'], tf.int32),
                        features['volume/shape'].values)
