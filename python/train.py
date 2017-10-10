import argparse
import tensorflow as tf

from mrtoct import model


def train(inputs_path, targets_path, chkpt_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  estimator = tf.contrib.gan.estimator.GANEstimator(
      model_dir=chkpt_path,
      generator_fn=model.gan.synthesis.generator_fn,
      discriminator_fn=model.gan.synthesis.discriminator_fn,
      generator_loss_fn=model.gan.synthesis.generator_loss_fn,
      discriminator_loss_fn=model.gan.synthesis.discriminator_loss_fn,
      generator_optimizer=tf.train.AdamOptimizer(
          params.learn_rate, params.beta1_rate),
      discriminator_optimizer=tf.train.AdamOptimizer(
          params.learn_rate, params.beta1_rate),
  )

  def input_fn():
    return model.train_patch_input_fn(
        inputs_path=inputs_path,
        targets_path=targets_path,
        inputs_shape=params.inputs_shape,
        targets_shape=params.targets_shape,
        volume_shape=params.volume_shape,
        batch_size=params.batch_size)

  estimator.train(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      batch_size=10,
      learn_rate=1e-6,
      beta1_rate=5e-1,
      inputs_shape=[32, 32, 32, 1],
      targets_shape=[16, 16, 16, 1],
      volume_shape=[260, 340, 360, 1])
  hparams.parse(args.hparams)

  train(inputs_path=args.inputs_path,
        targets_path=args.targets_path,
        chkpt_path=args.chkpt_path,
        params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--chkpt-path', default='results')
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
