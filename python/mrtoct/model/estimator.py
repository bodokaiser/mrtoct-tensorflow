import tensorflow as tf

from mrtoct import data, patch, ioutil
from mrtoct.model import losses


def model_fn(features, labels, mode, params):
  inputs, targets = features['inputs'], labels['targets']

  if params.data_format == 'channels_first':
    nchw_transform = data.transform.DataFormat2D('channels_first')
    nhwc_transform = data.transform.DataFormat2D('channels_last')

    outputs = params.generator_fn(nchw_transform(inputs), params.data_format)
    outputs = nhwc_transform(outputs)
  else:
    outputs = params.generator_fn(inputs)

  tf.summary.image('inputs', inputs, max_outputs=1)
  tf.summary.image('outputs', outputs, max_outputs=1)
  tf.summary.image('targets', targets, max_outputs=1)
  tf.summary.image('residue', targets - outputs, max_outputs=1)

  if tf.estimator.ModeKeys.PREDICT == mode:
    return tf.estimator.EstimatorSpec(mode, {'outputs': outputs})

  mse = tf.losses.mean_squared_error(targets, outputs)
  mae = tf.losses.absolute_difference(targets, outputs)
  gdl = losses.gradient_difference_loss_2d(targets, outputs)

  tf.summary.scalar('mean_squared_error', mse)
  tf.summary.scalar('mean_absolute_error', mae)
  tf.summary.scalar('gradient_difference_loss', gdl)

  loss = 3 * mae + gdl

  tf.summary.scalar('total_loss', loss)

  vars = tf.trainable_variables()

  gdl_grad = tf.global_norm(tf.gradients(gdl, vars))
  mae_grad = tf.global_norm(tf.gradients(mae, vars))

  tf.summary.scalar('gradient_difference_loss_gradient', gdl_grad)
  tf.summary.scalar('mean_absolute_error_gradient', mae_grad)

  if tf.estimator.ModeKeys.EVAL == mode:
    return tf.estimator.EstimatorSpec(
        mode, {'outputs': outputs}, loss)

  optimizer = tf.train.AdamOptimizer(params.learn_rate, params.beta1_rate)

  train = optimizer.minimize(loss, tf.train.get_global_step())

  return tf.estimator.EstimatorSpec(
      mode, {'outputs': outputs}, loss, train)


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
                    .map(post_transform)
                    .cache())

  targets_dataset = (tf.data
                     .TFRecordDataset(targets_path, ioutil.TFRecordCString)
                     .map(pre_transform)
                     .apply(tf.contrib.data.unbatch())
                     .map(post_transform)
                     .cache())

  dataset = (tf.data.Dataset
             .zip((inputs_dataset, targets_dataset))
             .batch(batch_size)
             .repeat())

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
