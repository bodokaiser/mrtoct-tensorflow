import argparse
import os

import tensorflow as tf

from mrtoct import data
from mrtoct import hook
from mrtoct import model
from mrtoct import ioutil

TEMPDIR = '/tmp/mrtoct'

def train(train_path, valid_path, result_path, num_epochs):
    train_dataset = tf.contrib.data.Dataset.zip((
        data.make_dataset(os.path.join(train_path, '*ct.tfrecord')),
        data.make_dataset(os.path.join(train_path, '*mr.tfrecord')),
    )).shuffle(1000).batch(16).repeat(num_epochs)
    valid_dataset = tf.contrib.data.Dataset.zip((
        data.make_dataset(os.path.join(valid_path, '*ct.tfrecord')),
        data.make_dataset(os.path.join(valid_path, '*mr.tfrecord')),
    )).batch(16).repeat()

    iterator = data.TrainValidIterator(train_dataset, valid_dataset)

    inputs, targets = iterator.get_next()
    outputs = model.unet(inputs)

    loss = tf.losses.mean_squared_error(targets, outputs)

    train = tf.train.GradientDescentOptimizer(1e-8).minimize(loss,
        tf.train.get_or_create_global_step())

    hooks = [
        hook.TrainValidFeedHook(iterator, 100),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        while not sess.should_stop():
            print(sess.run([loss, train]))

def convert(input_path, output_path):
    for e in os.listdir(input_path):
        source = os.path.join(input_path, e)

        if not os.path.isdir(source):
            continue

        for m in ['ct', 'mr']:
            target = os.path.join(output_path, f'{e}{m}.tfrecord')

            with tf.python_io.TFRecordWriter(target) as writer:
                volume = ioutil.read_nifti(os.path.join(source, f'{m}.nii'))

                for i in range(volume.shape[-1]):
                    writer.write(ioutil.encode_example(volume[:,:,i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='action')
    subparsers.requred = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train-path',
        default=os.path.join(TEMPDIR, 'testing'))
    parser_train.add_argument('--valid-path',
        default=os.path.join(TEMPDIR, 'training'))
    parser_train.add_argument('--result-path',
        default=os.path.join(TEMPDIR, 'results'))
    parser_train.add_argument('--num-epochs', type=int)

    parser_convert = subparsers.add_parser('convert')
    parser_convert.add_argument('--input-path', default='../data/nii')
    parser_convert.add_argument('--output-path', default='../data/tfrecord')

    args = parser.parse_args()

    if args.action == 'train':
        train(args.train_path, args.valid_path, args.result_path,
            args.num_epochs)
    elif args.action == 'convert':
        convert(args.input_path, args.output_path)
    else:
        parser.error(f'invalid action {args.action}')
