import tensorflow as tf

from mrtoct import data, patch, ioutil


compression = ioutil.TFRecordOptions.get_compression_type_string(
    ioutil.TFRecordOptions)


def train_slice_input_fn(inputs_path, targets_path, slice_shape, batch_size):
  transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.CastType(),
      data.transform.Normalize(),
      data.transform.CenterMean(),
      data.transform.CenterPad(slice_shape),
  ])

  inputs_dataset = tf.data.TFRecordDataset(
      inputs_path, compression).map(transform).cache()
  targets_dataset = tf.data.TFRecordDataset(
      targets_path, compression).map(transform).cache()

  dataset = tf.data.Dataset.zip(
      (inputs_dataset, targets_dataset)).batch(batch_size).repeat()

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
        data.transform.CastType(),
        data.transform.Normalize(),
        data.transform.CenterMean(),
        data.transform.CenterPad(volume_shape),
    ])

    inputs_volume_dataset = tf.data.TFRecordDataset(
        inputs_path, compression).map(volume_transform).cache()
    targets_volume_dataset = tf.data.TFRecordDataset(
        targets_path, compression).map(volume_transform).cache()

  with tf.name_scope('patch'):
    inputs_transform = data.transform.ExtractPatch(inputs_shape, index)
    targets_transform = data.transform.ExtractPatch(targets_shape, index)

    inputs_patch_dataset = inputs_volume_dataset.map(inputs_transform)
    targets_patch_dataset = targets_volume_dataset.map(targets_transform)

    patch_dataset = (tf.data.Dataset
                     .zip((inputs_patch_dataset, targets_patch_dataset))
                     .batch(batch_size))

    patch_iterator = patch_dataset.make_initializable_iterator()

    with tf.control_dependencies([patch_iterator.initializer]):
      return patch_iterator.get_next()
