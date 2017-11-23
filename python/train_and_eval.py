import os
import argparse
import tensorflow as tf

from mrtoct import model


def train(train_inputs_path, train_targets_path, eval_inputs_path,
          eval_targets_path, checkpoint_path, params):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

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

  def train_input_fn():
    return model.train_slice_input_fn(
        inputs_path=train_inputs_path,
        targets_path=train_targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)

  def eval_input_fn():
    return model.train_slice_input_fn(
        inputs_path=eval_inputs_path,
        targets_path=eval_targets_path,
        slice_height=params.slice_height,
        slice_width=params.slice_width,
        batch_size=params.batch_size)

  eval_hook = tf.train.SummarySaverHook(
      save_steps=10,
      scaffold=tf.train.Scaffold(),
      output_dir=os.path.join(checkpoint_path, 'eval'))

  while True:
    tf.logging.info('Training')
    estimator.train(train_input_fn, steps=1000)

    tf.logging.info('Validating')
    estimator.predict(eval_input_fn, hooks=[eval_hook])


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
