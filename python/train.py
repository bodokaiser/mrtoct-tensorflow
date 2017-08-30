import argparse
import os
import tensorflow as tf

from mrtoct import data
from mrtoct.model.generator import unet

TEMPDIR = '/tmp/mrtoct'

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.name_scope('data'):
        train_dataset = (data.make_zipped_dataset(args.train_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .shuffle(1000).batch(4)
            .repeat(args.num_epochs))
        valid_dataset = (data.make_zipped_dataset(args.valid_path)
            .filter(data.filter_nans)
            .filter(data.filter_incomplete)
            .batch(4).repeat())

        handle = tf.placeholder(tf.string, shape=[])

        iterator = data.make_iterator_from_handle(handle, train_dataset)
        train_iterator = train_dataset.make_one_shot_iterator()
        valid_iterator = valid_dataset.make_one_shot_iterator()

        handle_ops = [
            train_iterator.string_handle(),
            valid_iterator.string_handle(),
        ]

    with tf.name_scope('model'):
        inputs, targets = iterator.get_next()
        outputs = tf.layers.conv2d(inputs, 1, 3, padding='SAME')
        #outputs = unet.model(inputs)

        tf.summary.image('inputs', inputs)
        tf.summary.image('outputs', outputs)
        tf.summary.image('targets', targets)

    with tf.name_scope('loss'):
        mse_op = tf.losses.mean_squared_error(targets, outputs)
        loss_op = mse_op

        tf.summary.scalar('mse', mse_op)
        tf.summary.scalar('total', loss_op)

    with tf.name_scope('train'):
        mode = tf.placeholder(tf.string)
        step_op = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op, step_op)

    summary_op = tf.summary.merge_all()

    class FeedHook(tf.train.SessionRunHook):

        def __init__(self):
            self.summary = None
            self.writers = {
                'train': tf.summary.FileWriter(os.path.join(args.result_path,
                    'training'), tf.get_default_graph()),
                'valid': tf.summary.FileWriter(os.path.join(args.result_path,
                    'validation')),
            }

        def after_create_session(self, sess, coord):
            self.train_handle, self.valid_handle = sess.run(handle_ops)
            self.step = sess.run(step_op)

        def before_run(self, run_context):
            fetches = [step_op, summary_op]

            if self.step % 10 == 0:
                feed_dict = {handle: self.valid_handle, mode: 'valid'}
            else:
                feed_dict = {handle: self.train_handle, mode: 'train'}

            if self.summary is not None:
                self.writers[feed_dict[mode]].add_summary(
                    self.summary, self.step)

            return tf.train.SessionRunArgs(fetches, feed_dict)

        def after_run(self, run_context, run_values):
            self.step, self.summary = run_values.results

    config = dict(
        hooks=[
            FeedHook(),
            tf.train.NanTensorHook(loss_op),
            tf.train.LoggingTensorHook({
                'loss': loss_op,
                'mode': mode,
            }, every_n_secs=120),
        ],
        save_checkpoint_secs=600,
        save_summaries_secs=None,
        checkpoint_dir=args.result_path,
    )

    with tf.train.MonitoredTrainingSession(**config) as sess:
        while not sess.should_stop():
            sess.run(train_op)

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
