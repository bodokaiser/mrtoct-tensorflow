import argparse
import numpy as np
import tensorflow as tf

from mrtoct import ioutil, model


def predict(inputs_path, outputs_path, checkpoint_path, params):
  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions

  estimator = tf.estimator.Estimator(
      model_fn=model.gan_model_fn,
      params=params)

  def create_input_fn(offset):
    def input_fn():
      index, inputs = model.predict_patch_input_fn(
          offset=offset,
          delta=params.delta,
          inputs_path=inputs_path,
          volume_shape=params.volume_shape,
          inputs_shape=params.inputs_shape,
          batch_size=params.batch_size)

      return {'inputs': inputs, 'index': index}
    return input_fn

  predictions = estimator.predict(
      input_fn=create_input_fn(0),
      checkpoint_path=checkpoint_path)

  outputs = [p['outputs'] for p in predictions]

  print('foo')
  print(outputs)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      delta=5,
      batch_size=10,
      inputs_shape=[32, 32, 32],
      volume_shape=[260, 340, 360, args.iteration],
      data_format='channels_last',
      generator_fn=model.synthesis.generator_fn)
  hparams.parse(args.hparams)

  predict(inputs_path=args.inputs_path,
          outputs_path=args.outputs_path,
          checkpoint_path=args.checkpoint_path,
          params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('predict')
  parser.add_argument('--iteration', type=int, required=True)
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
