import argparse
import os
import tempfile

import tensorflow as tf

from mrtoct import model
from mrtoct import ioutil
from mrtoct import loss

def listdirs(path):
    isdir = lambda f: os.path.isdir(os.path.join(path, f))
    return list(filter(isdir, os.listdir(path)))

TEMPDIR = '/tmp/mrtoct'

def train(testdir, traindir, resultdir):
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    handle = tf.placeholder(tf.string, shape=[])

    test_records = [os.path.join(testdir, f) for f in os.listdir(testdir)]
    train_records = [os.path.join(traindir, f) for f in os.listdir(traindir)]

    test_dataset = tf.contrib.data.TFRecordDataset(test_records).map(
        ioutil.decode_example)
    train_dataset = tf.contrib.data.TFRecordDataset(train_records).map(
        ioutil.decode_example)

    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
        train_dataset.output_types, train_dataset.output_shapes)

    next_element = iterator.get_next()

    test_iterator = test_dataset.make_one_shot_iterator()
    train_iterator = train_dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        test_handle = sess.run(test_iterator.string_handle())
        train_handle = sess.run(train_iterator.string_handle())

        for i in range(20):
            inputs, targets = sess.run(next_element, feed_dict={
                handle: train_handle})
            outputs = model.unet(inputs)

            print(model.mse(outputs, targets))


def convert(inputdir, outputdir):
    for d in listdirs(inputdir):
        filename = os.path.join(outputdir, f'{d}.tfrecord')

        ct = ioutil.read_nifti(os.path.join(inputdir, d, 'ct.nii'))
        mr = ioutil.read_nifti(os.path.join(inputdir, d, 'mr.nii'))

        with tf.python_io.TFRecordWriter(filename) as writer:
            assert ct.shape[-1] == mr.shape[-1]

            for i in range(ct.shape[-1]):
                writer.write(ioutil.encode_example(ct[:,:,i], mr[:,:,i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='action')
    subparsers.requred = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--testdir',
        default=os.path.join(TEMPDIR, 'testing'))
    parser_train.add_argument('--traindir',
        default=os.path.join(TEMPDIR, 'training'))
    parser_train.add_argument('--resultdir',
        default=os.path.join(TEMPDIR, 'results'))

    parser_convert = subparsers.add_parser('convert')
    parser_convert.add_argument('--inputdir', default='../data/nii')
    parser_convert.add_argument('--outputdir', default='../data/tfrecord')

    args = parser.parse_args()

    if args.action == 'train':
        train(args.testdir, args.traindir, args.resultdir)
    elif args.action == 'convert':
        convert(args.inputdir, args.outputdir)
    else:
        parser.error(f'invalid action {args.action}')
