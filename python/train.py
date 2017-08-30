import argparse
import os
import tensorflow as tf

from mrtoct import data
from mrtoct.model import estimator
from mrtoct.model.generator import unet

TEMPDIR = '/tmp/mrtoct'

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('data'):
        train_dataset = (data.make_zipped_dataset(args.train_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .shuffle(2000).batch(10)
            .repeat(args.num_epochs))
        valid_dataset = (data.make_zipped_dataset(args.valid_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .batch(10).repeat())

    experiment = tf.contrib.learn.Experiment(
        estimator=tf.estimator.Estimator(
            model_fn=estimator.model_fn,
            model_dir=args.result_path,
            config = tf.contrib.learn.RunConfig(
                model_dir=args.result_path),
            params={'lr': 1e-3}),
        train_input_fn=estimator.make_input_fn(train_dataset),
        eval_input_fn=estimator.make_input_fn(valid_dataset))
    experiment.train_and_evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path',
        default=os.path.join(TEMPDIR, 'training'))
    parser.add_argument('--valid-path',
        default=os.path.join(TEMPDIR, 'validation'))
    parser.add_argument('--result-path',
        default=os.path.join(TEMPDIR, 'results'))
    parser.add_argument('--num-epochs', type=int)

    main(parser.parse_args())
