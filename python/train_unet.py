import argparse
import tensorflow as tf

from mrtoct import model


def train(inputs_path, targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.log_device_placement = True

  estimator = tf.estimator.Estimator(
      model_fn=model.model_fn,
      model_dir=checkpoint_path,
      params=params)

  def input_fn():
    inputs, targets = model.train_slice_input_fn(
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
      data_format='channels_first',
      generator_fn=model.unet.generator_fn)
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
