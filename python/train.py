import argparse
import os
import tensorflow as tf

from mrtoct import data
from mrtoct.model import estimator
from mrtoct.model.generator import unet, dummy

TEMPDIR = '/tmp/mrtoct'

def main(train_path, valid_path, result_path, params, batch_size, num_epochs):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('data'):
        train_dataset = (data.make_zipped_dataset(train_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .shuffle(2000).batch(batch_size)
            .repeat(num_epochs))
        valid_dataset = (data.make_zipped_dataset(valid_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .batch(batch_size).repeat())

    experiment = tf.contrib.learn.Experiment(
        estimator=tf.estimator.Estimator(
            model_fn=estimator.model_fn,
            model_dir=args.result_path,
            config = tf.contrib.learn.RunConfig(
                model_dir=args.result_path),
            params=params),
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
    parser.add_argument('--generator',
        choices=['unet', 'dummy'], default='dummy')
    parser.add_argument('--discriminator',
        choices=['pixel', 'cnn'])
    parser.add_argument('--num-epochs',
        type=int, default=0)
    parser.add_argument('--batch-size',
        type=int, default=10)
    parser.add_argument('--learn-rate',
        type=float, default=1e-3)

    args = parser.parse_args()

    if args.generator == 'unet':
        gen = unet.model
    elif args.generator == 'dummy':
        gen = dummy.model
    else:
        parser.error(f'unknown generator {args.generator}')

    if args.discriminator == 'pixel':
        dis = pixel.model
    elif args.discriminator == 'cnn':
        dis = cnn.model
    else:
        dis = None

    main(args.train_path, args.valid_path, args.result_path, {
        'generator': gen, 'discriminator': dis, 'lr': args.learn_rate,
    }, args.batch_size, args.num_epochs)