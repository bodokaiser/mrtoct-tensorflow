import tensorflow as tf

from mrtoct import ioutil, data, patch


def train_slice_input_fn(inputs_path, targets_path, slice_shape, batch_size):
  pre_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
  ])
  post_transform = data.transform.Compose([
      data.transform.CropOrPad2D(*slice_shape),
      data.transform.ExpandDims(),
  ])

  inputs_dataset = (tf.data
                    .TFRecordDataset(inputs_path, ioutil.TFRecordCString)
                    .map(pre_transform)
                    .apply(tf.contrib.data.unbatch())
                    .map(post_transform))

  targets_dataset = (tf.data
                     .TFRecordDataset(targets_path, ioutil.TFRecordCString)
                     .map(pre_transform)
                     .apply(tf.contrib.data.unbatch())
                     .map(post_transform))

  dataset = (tf.data.Dataset
             .zip((inputs_dataset, targets_dataset))
             .batch(batch_size)
             .repeat())

  return dataset.make_one_shot_iterator().get_next()


def predict_slice_input_fn(inputs_path, slice_shape, offset):
  pre_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
  ])
  post_transform = data.transform.Compose([
      data.transform.CropOrPad2D(*slice_shape),
      data.transform.ExpandDims(),
  ])

  dataset = (tf.data
             .TFRecordDataset(inputs_path, ioutil.TFRecordCString)
             .skip(offset)
             .take(1)
             .map(pre_transform)
             .apply(tf.contrib.data.unbatch())
             .map(post_transform)
             .batch(1))

  return dataset.make_one_shot_iterator().get_next()


def train_patch_input_fn(inputs_path, targets_path, volume_shape, inputs_shape,
                         targets_shape, batch_size):
  with tf.name_scope('sample'):
    offset = tf.convert_to_tensor(inputs_shape[:3]) // 2
    length = tf.convert_to_tensor(volume_shape[:3]) - offset

    index = patch.sample_uniform_3d(offset, length, 1)[0]

  with tf.name_scope('volume'):
    volume_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.Normalize(),
        data.transform.CenterPad3D(*volume_shape[:3]),
        data.transform.Lambda(lambda x: tf.reshape(x, volume_shape)),
    ])

    inputs_volume_dataset = tf.data.TFRecordDataset(
        inputs_path, ioutil.TFRecordCString).map(volume_transform).cache()
    targets_volume_dataset = tf.data.TFRecordDataset(
        targets_path, ioutil.TFRecordCString).map(volume_transform).cache()

  with tf.name_scope('patch'):
    inputs_transform = data.transform.IndexCrop3D(inputs_shape, index)
    targets_transform = data.transform.IndexCrop3D(targets_shape, index)

    inputs_patch_dataset = inputs_volume_dataset.map(inputs_transform)
    targets_patch_dataset = targets_volume_dataset.map(targets_transform)

    patch_dataset = (tf.data.Dataset
                     .zip((inputs_patch_dataset, targets_patch_dataset))
                     .batch(batch_size))

    patch_iterator = patch_dataset.make_initializable_iterator()

    with tf.control_dependencies([patch_iterator.initializer]):
      return patch_iterator.get_next()
