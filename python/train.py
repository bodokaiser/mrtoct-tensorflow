import argparse
import os
import tensorflow as tf

from mrtoct import data
from mrtoct import model

TEMPDIR = '/tmp/mrtoct'

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('data'):
        train_dataset = (data.make_zipped_dataset(args.train_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .shuffle(1000).batch(16)
            .repeat(args.num_epochs))
        valid_dataset = (data.make_zipped_dataset(args.valid_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .batch(16).repeat())

        iterator = data.TrainValidIterator(train_dataset, valid_dataset)

        handle = iterator.get_handle()
        handle_ops = [
            iterator.get_train_handle(),
            iterator.get_valid_handle(),
        ]

    with tf.name_scope('model'):
        inputs, targets = iterator.get_next()
        outputs = tf.layers.conv2d(inputs, 1, 3, padding='SAME')

        tf.summary.image('inputs', inputs)
        tf.summary.image('outputs', outputs)
        tf.summary.image('targets', targets)

    with tf.name_scope('loss'):
        mse_op = tf.losses.mean_squared_error(targets, outputs)
        loss_op = mse_op

        tf.summary.scalar('mse', mse_op)
        tf.summary.scalar('total', loss_op)

    with tf.name_scope('train'):
        step_op = tf.train.get_or_create_global_step()
        train_op = tf.train.GradientDescentOptimizer(1e-10).minimize(loss_op,
            step_op)

    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(
        os.path.join(args.result_path, 'training'), tf.get_default_graph())
    valid_writer = tf.summary.FileWriter(
        os.path.join(args.result_path, 'validation'))

    with tf.train.MonitoredTrainingSession() as sess:
        train_handle, valid_handle = sess.run(handle_ops)

        while not sess.should_stop():
            for i in range(10):
                fetches = [loss_op, train_op, summary_op, step_op]
                loss, _, summary, step = sess.run(fetches,
                    feed_dict={handle: train_handle})

                tf.logging.info(f'training: step {step}, loss {loss}')
                train_writer.add_summary(summary, step)

            fetches = [loss_op, train_op, summary_op, step_op]
            loss, _, summary, step = sess.run(fetches,
                feed_dict={handle: valid_handle})

            tf.logging.info(f'validation: step {step}, loss {loss}')
            valid_writer.add_summary(summary, step)

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