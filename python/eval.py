import argparse
import tensorflow as tf

from mrtoct import model


def eval(inputs_path, targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  estimator = tf.estimator.Estimator(
      model_fn=model.model_fn,
      model_dir=checkpoint_path,
      params=params)

  def input_fn():
    inputs, targets = model.eval_slice_input_fn(
        inputs_path=inputs_path,
        targets_path=targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)
    return {'inputs': inputs}, {'targets': targets}

  estimator.eval(input_fn)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      batch_size=16,
      slice_height=384,
      slice_width=384,
      generator_fn=model.unet.generator_fn)
  hparams.parse(args.hparams)

  eval(inputs_path=args.inputs_path,
       targets_path=args.targets_path,
       checkpoint_path=args.checkpoint_path,
       params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('train', description='''
    Evals model on tfrecords and saves outputs as tfrecord.
  ''')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--checkpoint-path', default='results')
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
