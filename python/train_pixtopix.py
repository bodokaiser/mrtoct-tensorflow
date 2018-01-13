import argparse
import tensorflow as tf

from mrtoct import model

INPUTS_MAX = 5200
TARGETS_MAX = 3700


def train(inputs_path, targets_path, checkpoint_path, params):
  estimator = tf.estimator.Estimator(
      model_fn=model.gan_model_fn,
      model_dir=checkpoint_path,
      params=params)

  def input_fn():
    inputs, targets = model.train_slice_input_fn(
        inputs_div=INPUTS_MAX,
        targets_div=TARGETS_MAX,
        inputs_path=inputs_path,
        targets_path=targets_path,
        slice_shape=params.slice_shape,
        batch_size=params.batch_size)
    return {'inputs': inputs}, {'targets': targets}

  estimator.train(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      learn_rate=2e-4,
      beta1_rate=5e-1,
      batch_size=16,
      slice_shape=[384, 384],
      weight_factor=1e-2,
      data_format='channels_first',
      generator_fn=model.pixtopix.generator_fn,
      discriminator_fn=model.pixtopix.discriminator_fn,
      generator_loss_fn=tf.contrib.gan.losses.least_squares_generator_loss,
      discriminator_loss_fn=tf.contrib.gan.losses.least_squares_discriminator_loss)
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
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
