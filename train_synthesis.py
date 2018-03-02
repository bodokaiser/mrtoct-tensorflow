import argparse
import tensorflow as tf

from mrtoct import model


def train(inputs_path, targets_path, checkpoint_path, params):
  estimator = tf.estimator.Estimator(
      model_fn=model.gan_model_fn,
      model_dir=checkpoint_path,
      params=params)

  def input_fn():
    inputs, targets = model.train_patch_input_fn(
        inputs_path=inputs_path,
        targets_path=targets_path,
        volume_shape=params.volume_shape,
        inputs_shape=params.inputs_shape,
        targets_shape=params.targets_shape,
        batch_size=params.batch_size)

    return {'inputs': inputs}, {'targets': targets}

  estimator.train(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      learn_rate=1e-6,
      beta1_rate=5e-1,
      batch_size=10,
      inputs_shape=[32, 32, 32],
      targets_shape=[16, 16, 16],
      volume_shape=[260, 340, 360, args.iteration],
      weight_factor=0.5,
      data_format='channels_last',
      generator_fn=model.synthesis.generator_fn,
      discriminator_fn=model.synthesis.discriminator_fn,
      generator_loss_fn=tf.contrib.gan.losses.modified_generator_loss,
      discriminator_loss_fn=tf.contrib.gan.losses.modified_discriminator_loss)
  hparams.parse(args.hparams)

  train(inputs_path=args.inputs_path,
        targets_path=args.targets_path,
        checkpoint_path=args.checkpoint_path,
        params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('train')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)
  parser.add_argument('--iteration', type=int, default=1)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
