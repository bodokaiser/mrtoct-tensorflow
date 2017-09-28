import argparse
import os
import tensorflow as tf

from mrtoct import ioutil, data, model, patch


def train(input_path, output_path, params, batch_size, num_epochs):
  inputs_filenames = tf.gfile.Glob(
      os.path.join(input_path, 're-co-mr-*.tfrecord'))
  targets_filenames = tf.gfile.Glob(
      os.path.join(input_path, 're-ct-*.tfrecord'))

  if len(inputs_filenames) != len(targets_filenames):
    raise RuntimeError('input and target volumes do not match')

  tf.logging.info(f'found {len(inputs_filenames)} volume pairs to train')

  with tf.name_scope('config'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    vshape = tf.convert_to_tensor(params.volume_shape)
    pshape = tf.convert_to_tensor(params.patch_shape)

  with tf.name_scope('indices'):
    off = pshape // 2
    size = vshape - off

    indices = tf.concat([
        patch.sample_meshgrid_3d(off, size, params.sample_delta),
        patch.sample_uniform_3d(off, size, params.sample_num),
    ], -1)
    indices = tf.random_shuffle(indices)
    indices_len = tf.to_int64(tf.shape(indices)[0])

  with tf.name_scope('dataset'):
    options = ioutil.TFRecordOptions.get_compression_type_string(
        ioutil.TFRecordOptions)

    volume_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.CastType(),
        data.transform.Normalize(),
        data.transform.CenterPad(vshape),
        data.transform.CenterMean(),
    ])
    patch_transform = data.transform.Compose([
        data.transform.ExtractPatch(pshape),
        data.transform.ExpandDims(),
    ])

    with tf.name_scope('index'):
      index_dataset = data.Dataset.from_tensor_slices(indices)

    with tf.name_scope('volume'):
      inputs_volume_dataset = (data.TFRecordDataset(inputs_filenames, options)
                                   .map(volume_transform).cache())
      targets_volume_dataset = (data.TFRecordDataset(inputs_filenames, options)
                                    .map(volume_transform).cache())

    def extract_patches(volume):
      volume_dataset = data.Dataset.from_tensors(volume).repeat(indices_len)

      return (data.Dataset
              .zip((index_dataset, volume_dataset))
              .map(patch_transform))

    with tf.name_scope('patch'):
      inputs_patch_dataset = inputs_volume_dataset.flat_map(extract_patches)
      targets_patch_dataset = targets_volume_dataset.flat_map(extract_patches)

      patch_dataset = (data.Dataset
                       .zip((inputs_patch_dataset, targets_patch_dataset))
                       .batch(params.batch_size)
                       .repeat(params.num_epochs))

  with tf.name_scope('iterator'):
    patch_iterator = patch_dataset.make_initializable_iterator()

    input_patches, target_patches = patch_iterator.get_next()

  with tf.name_scope('model'):
    spec = model.create_generative_adversarial_network(
        input_patches, target_patches, model.Mode.TRAIN, params)

    step = tf.train.get_global_step()

    output_patches = spec.outputs

    deprocess = data.transform.UncenterMean()

    tf.summary.image('input', deprocess(input_patches[:, :, :, 0]), 1)
    tf.summary.image('output', deprocess(output_patches[:, :, :, 0]), 1)
    tf.summary.image('target', deprocess(target_patches[:, :, :, 0]), 1)

    scaffold = tf.train.Scaffold(init_op=tf.group(
        tf.global_variables_initializer(), patch_iterator.initializer))

  tf.logging.info('finalized computation graph, starting training session')

  with tf.train.MonitoredTrainingSession(
          config=config,
          scaffold=scaffold,
          checkpoint_dir=output_path,
          save_checkpoint_secs=600,
          save_summaries_secs=100) as sess:
    num_indices = sess.run(indices_len)

    while not sess.should_stop():
      s, _ = sess.run([step, spec.train_op])

      if s % (num_indices // params.batch_size - 1) == 0:
        # reininitialize patch iterator in order to get sample new indices
        sess.run(patch_iterator.initializer)

      if s % 100 == 0:
        tf.logging.info(f'step: {s}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      learn_rate=1e-6,
      beta1_rate=5e-1,
      mse_weight=1.00,
      mae_weight=0.00,
      gdl_weight=1.00,
      adv_weight=0.50,
      sample_delta=4,
      sample_num=10000,
      patch_shape=[32, 32, 32],
      volume_shape=[240, 300, 340],
      networks=model.gan.synthesis)
  hparams.parse(args.hparams)

  train(args.input_path, args.output_path, hparams,
        args.batch_size, args.num_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', default='results')
  parser.add_argument('--num-epochs', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=10)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
