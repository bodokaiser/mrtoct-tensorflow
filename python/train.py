import argparse
import tensorflow as tf

from mrtoct import ioutil, data, model, patch


def train(inputs_path, targets_path, log_path, params, batch_size, num_epochs):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.name_scope('shapes'):
    vshape = tf.convert_to_tensor(params.volume_shape, name='volume_shape')
    pshape = tf.convert_to_tensor(params.patch_shape, name='patch_shape')

  with tf.name_scope('indices'):
    off = pshape[:3] // 2
    size = vshape[:3] - off

    indices = patch.sample_uniform_3d(off, size, params.sample_num)

  with tf.name_scope('dataset'):
    options = ioutil.TFRecordOptions.get_compression_type_string(
        ioutil.TFRecordOptions)

    volume_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.CastType(),
        data.transform.Normalize(),
        data.transform.CenterMean(),
        data.transform.CenterPad(vshape),
    ])

    with tf.name_scope('index'):
      index_dataset = data.Dataset.from_tensor_slices(indices)

    with tf.name_scope('volume'):
      inputs_volume_dataset = data.TFRecordDataset(
          inputs_path, options).map(volume_transform).cache()
      targets_volume_dataset = data.TFRecordDataset(
          targets_path, options).map(volume_transform).cache()

    def patch_extractor(shape):
      patch_transform = data.transform.ExtractPatch(shape)

      def extract_patches(volume):
        volume_dataset = (data.Dataset.from_tensors(volume)
                          .repeat(params.sample_num))

        return (data.Dataset
                .zip((index_dataset, volume_dataset))
                .map(patch_transform))

      return extract_patches

    with tf.name_scope('patch'):
      inputs_patch_dataset = inputs_volume_dataset.flat_map(
          patch_extractor([32, 32, 32, 1]))
      targets_patch_dataset = targets_volume_dataset.flat_map(
          patch_extractor([16, 16, 16, 1]))

      patch_dataset = (data.Dataset
                       .zip((inputs_patch_dataset, targets_patch_dataset))
                       .batch(batch_size)
                       .repeat(num_epochs))

  with tf.name_scope('iterator'):
    patch_iterator = patch_dataset.make_initializable_iterator()

    with tf.control_dependencies([patch_iterator.initializer]):
      input_patches, target_patches = patch_iterator.get_next()

  with tf.name_scope('model'):
    spec = model.create_generative_adversarial_network(
        input_patches, target_patches, model.Mode.TRAIN, params)

    step = tf.train.get_global_step()

    output_patches = spec.outputs

    def deprocess(x):
      transform = data.transform.UncenterMean()

      return tf.image.convert_image_dtype(transform(x[:, :, :, 0]),
                                          dtype=tf.uint8)

    tf.summary.image('input', deprocess(input_patches), 1)
    tf.summary.image('output', deprocess(output_patches), 1)
    tf.summary.image('target', deprocess(target_patches), 1)

  tf.logging.info('finalized computation graph, starting training session')

  with tf.train.MonitoredTrainingSession(
          config=config,
          checkpoint_dir=log_path,
          save_checkpoint_secs=600,
          save_summaries_secs=100) as sess:
    while not sess.should_stop():
      s, _ = sess.run([step, spec.train_op])

      if s % (params.sample_num // batch_size - 1) == 0:
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
      sample_num=7000,
      patch_shape=[32, 32, 32, 1],
      volume_shape=[260, 340, 360, 1],
      generator=model.gan.synthesis.generator_network,
      discriminator=model.gan.synthesis.discriminator_network)
  hparams.parse(args.hparams)

  train(args.inputs_path, args.targets_path, args.log_path, hparams,
        args.batch_size, args.num_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--log-path', default='results')
  parser.add_argument('--num-epochs', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=10)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
