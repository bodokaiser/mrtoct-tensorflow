import argparse
import tensorflow as tf

from mrtoct import model


def train(inputs_path, targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  estimator = tf.contrib.gan.estimator.GANEstimator(
      add_summaries=None,
      use_loss_summaries=False,
      model_dir=checkpoint_path,
      generator_fn=model.synthesis.generator_fn,
      discriminator_fn=model.synthesis.discriminator_fn,
      generator_loss_fn=model.synthesis.generator_loss_fn,
      discriminator_loss_fn=model.synthesis.discriminator_loss_fn,
      generator_optimizer=tf.train.AdamOptimizer(params.learn_rate,
                                                 params.beta1_rate),
      discriminator_optimizer=tf.train.AdamOptimizer(params.learn_rate,
                                                     params.beta1_rate))

  def input_fn():
    return model.train_patch_input_fn(
        inputs_path=inputs_path,
        targets_path=targets_path,
        volume_shape=params.volume_shape,
        inputs_shape=params.inputs_shape,
        targets_shape=params.targets_shape,
        batch_size=params.batch_size)

  estimator.train(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      learn_rate=1e-6,
      beta1_rate=5e-1,
      batch_size=10,
      inputs_shape=[32, 32, 32],
      targets_shape=[16, 16, 16],
      volume_shape=[260, 340, 360, args.iteration])
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
