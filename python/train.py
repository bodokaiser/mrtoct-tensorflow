import argparse
import os
import tensorflow as tf

from mrtoct.data import make_zipped_dataset
from mrtoct.data import filter_nans, filter_incomplete
from mrtoct.model import estimator
from mrtoct.model.generator import unet, dummy
from mrtoct.model.discriminator import cnn, pixel

TEMPDIR = '/tmp/mrtoct'

def main(train_path, valid_path, result_path, params, batch_size, num_epochs):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('dataset'):
        train_dataset = (make_zipped_dataset(train_path)
            .filter(filter_nans)
            .filter(filter_incomplete)
            .shuffle(2000).batch(batch_size)
            .repeat(num_epochs))
        valid_dataset = (make_zipped_dataset(valid_path)
            .filter(filter_nans)
            .filter(filter_incomplete)
            .batch(batch_size).repeat())

    with tf.name_scope('iterator'):
        iterator = tf.contrib.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(train_dataset)
        valid_init_op = iterator.make_initializer(valid_dataset)

    inputs, targets = iterator.get_next()

    step_op = tf.train.get_or_create_global_step()

    train_op = estimator.model_fn({'inputs': inputs}, {'targets': targets},
        tf.estimator.ModeKeys.TRAIN, params).train_op

    summary_op = tf.summary.merge_all()

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=result_path,
        save_summaries_steps=None,
        save_summaries_secs=None) as sess:
        train_writer = tf.summary.FileWriter(
            os.path.join(result_path, 'training'), sess.graph)
        valid_writer = tf.summary.FileWriter(
            os.path.join(result_path, 'validation'))

        while not sess.should_stop():
            sess.run(train_init_op)
            for i in range(50):
                step, summary, _ = sess.run([step_op, summary_op, train_op])
                train_writer.add_summary(summary, step)

            sess.run(valid_init_op)
            for i in range(5):
                step, summary, _ = sess.run([step_op, summary_op, train_op])
                valid_writer.add_summary(summary, step)

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
        type=int, default=None)
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