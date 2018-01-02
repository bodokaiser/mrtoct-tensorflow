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
      inputs = model.predict_slice_input_fn(
          offset=offset,
          slice_shape=params.slice_shape,
          inputs_path=inputs_path)

      return {'inputs': inputs}
    return input_fn

  with tf.python_io.TFRecordWriter(outputs_path, options) as writer:
    offset = 0

    while True:
      predictions = estimator.predict(
          input_fn=create_input_fn(offset),
          checkpoint_path=checkpoint_path)

      outputs = [p['outputs'] for p in predictions]

      if len(outputs) == 0:
        break
      else:
        offset += 1

      volume = np.stack(outputs, axis=0)
      volume *= float(np.iinfo(np.int16).max)

      writer.write(encoder.encode(volume))


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      data_format='channels_last',
      slice_shape=[384, 384],
      generator_fn=model.pixtopix.generator_fn,
      discriminator_fn=model.pixtopix.discriminator_fn)
  hparams.parse(args.hparams)

  predict(inputs_path=args.inputs_path,
          outputs_path=args.outputs_path,
          checkpoint_path=args.checkpoint_path,
          params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('predict')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
