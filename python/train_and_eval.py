import os
import argparse
import tensorflow as tf

from mrtoct import model


def train(train_inputs_path, train_targets_path, eval_inputs_path,
          eval_targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  '''
  estimator = tf.contrib.gan.estimator.GANEstimator(
      add_summaries=None,
      use_loss_summaries=False,
      model_dir=checkpoint_path,
      generator_fn=model.pixtopix.generator_fn,
      discriminator_fn=model.pixtopix.discriminator_fn,
      generator_loss_fn=model.pixtopix.generator_loss_fn,
      discriminator_loss_fn=model.pixtopix.discriminator_loss_fn,
      generator_optimizer=tf.train.AdamOptimizer(params.lr, params.beta1),
      discriminator_optimizer=tf.train.AdamOptimizer(params.lr, params.beta1))
  '''

  estimator = tf.estimator.Estimator(
      model_fn=model.model_fn,
      model_dir=checkpoint_path,
      params=params)

  def train_input_fn():
    inputs, targets = model.train_slice_input_fn(
        inputs_path=train_inputs_path,
        targets_path=train_targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)
    return {'inputs': inputs}, {'targets': targets}

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)

  def eval_input_fn():
    inputs, targets = model.train_slice_input_fn(
        inputs_path=eval_inputs_path,
        targets_path=eval_targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)
    return {'inputs': inputs}, {'targets': targets}

  eval_hook = tf.train.SummarySaverHook(
      save_steps=10,
      scaffold=tf.train.Scaffold(),
      output_dir=os.path.join(checkpoint_path, 'eval'))
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, hooks=[eval_hook])

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      lr=2e-4,
      beta1=5e-1,
      batch_size=16,
      slice_height=384,
      slice_width=384,
      generator_fn=model.unet.generator_fn)
  hparams.parse(args.hparams)

  train(train_inputs_path=args.train_inputs_path,
        train_targets_path=args.train_targets_path,
        eval_inputs_path=args.eval_inputs_path,
        eval_targets_path=args.eval_targets_path,
        checkpoint_path=args.checkpoint_path,
        params=hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('train', description='''
    Trains model on tfrecords and saves checkpoints with events.
  ''')
  parser.add_argument('--train-inputs-path', required=True)
  parser.add_argument('--train-targets-path', required=True)
  parser.add_argument('--eval-inputs-path', required=True)
  parser.add_argument('--eval-targets-path', required=True)
  parser.add_argument('--checkpoint-path', default='results')
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
