import argparse
import os
import tensorflow as tf

from mrtoct import ioutil, data, model, patch


def train(inputs_path, targets_path, params, log_path, batch_size, num_epochs):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.name_scope('dataset'):
    compression = ioutil.TFRecordOptions.get_compression_type_string(
        ioutil.TFRecordOptions)

    patch_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.CastType(),
        data.transform.Normalize(),
        data.transform.CenterMean(),
    ])

    with tf.name_scope('inputs'):
      inputs_dataset = data.TFRecordDataset(
          inputs_path, compression).map(patch_transform).cache()

    with tf.name_scope('targets'):
      targets_dataset = data.TFRecordDataset(
          targets_path, compression).map(patch_transform).cache()

    with tf.name_scope('patches'):
      patch_dataset = (data.Dataset
                       .zip((inputs_dataset, targets_dataset))
                       .batch(batch_size)
                       .repeat(num_epochs))

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
          checkpoint_dir=log_path,
          save_checkpoint_secs=600,
          save_summaries_secs=100) as sess:

    while not sess.should_stop():
      s, _ = sess.run([step, spec.train_op])

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
      generator=model.gan.synthesis.generator_network,
      discriminator=model.gan.synthesis.discriminator_network)
  hparams.parse(args.hparams)

  train(args.input_path, args.output_path, hparams, args.log_path,
        args.batch_size, args.num_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--log-path', default='results')
  parser.add_argument('--num-epochs', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=10)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
