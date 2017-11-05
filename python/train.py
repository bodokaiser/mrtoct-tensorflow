import argparse
import tensorflow as tf

from mrtoct import model


def train(inputs_path, targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  estimator = tf.contrib.gan.estimator.GANEstimator(
      model_dir=checkpoint_path,
      generator_fn=model.pixtopix.generator_fn,
      discriminator_fn=model.pixtopix.discriminator_fn,
      generator_loss_fn=model.pixtopix.generator_loss_fn,
      discriminator_loss_fn=model.pixtopix.discriminator_loss_fn,
      generator_optimizer=model.pixtopix.optimizer_fn(),
      discriminator_optimizer=model.pixtopix.optimizer_fn())

  def input_fn():
    return model.train_slice_input_fn(
        inputs_path=inputs_path,
        targets_path=targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)

  estimator.train(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      batch_size=10,
      slice_height=288,
      slice_width=288)
  hparams.parse(args.hparams)

  train(inputs_path=args.inputs_path,
        targets_path=args.targets_path,
        checkpoint_path=args.checkpoint_path,
        params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('train', description='''
    Trains model on tfrecords and saves checkpoints with events.
  ''')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--checkpoint-path', default='results')
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
