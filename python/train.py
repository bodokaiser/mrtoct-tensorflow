import argparse
import os
import tensorflow as tf

from mrtoct import data
from mrtoct import model


def train(train_path, valid_path, result_path, params, batch_size, num_epochs):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('dataset'):
        train_dataset = (data.make_zipped_dataset(train_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .shuffle(2000).batch(batch_size)
            .repeat(num_epochs))
        valid_dataset = (data.make_zipped_dataset(valid_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .batch(batch_size).repeat())

    with tf.name_scope('iterator'):
        iterator = tf.contrib.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(train_dataset)
        valid_init_op = iterator.make_initializer(valid_dataset)

    inputs, targets = iterator.get_next()

    spec = model.create_generative_adversarial_network(inputs, targets,
        model.Mode.TRAIN, params)

    fetches = [tf.train.get_or_create_global_step(),
        spec.summary_op, spec.train_op]

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
                step, summary, _ = sess.run(fetches)
                train_writer.add_summary(summary, step)

            sess.run(valid_init_op)
            for i in range(5):
                step, summary, _ = sess.run(fetches)
                valid_writer.add_summary(summary, step)


def main(args):
    hparams = tf.contrib.training.HParams(
        learn_rate=2e-4,
        beta1_rate=5e-1,
        mse_weight=0.00,
        mae_weight=1.00,
        loss_weight=100,
        num_filters=64)
    hparams.parse(args.hparams)

    train(args.train_path, args.valid_path, args.result_path, hparams,
        args.batch_size, args.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = subparsers.add_parser('train')
    parser.add_argument('--train-path', default='training')
    parser.add_argument('--valid-path', default='validation')
    parser.add_argument('--result-path', default='results')
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--hparams', type=str, default='')

    main(parser.parse_args())