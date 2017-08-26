import argparse
import os
import tempfile

import tensorflow as tf

from mrtoct import model
from mrtoct import ioutil
from mrtoct import losses

from matplotlib import pyplot as plt

def listdirs(path):
    isdir = lambda f: os.path.isdir(os.path.join(path, f))
    return list(filter(isdir, os.listdir(path)))

TEMPDIR = '/tmp/mrtoct'

def process(image):
    # normalize to [0, 1]
    image = tf.subtract(image, tf.reduce_min(image))
    image = tf.divide(image, tf.reduce_max(image))
    # normalize to [-1,+1]
    image = tf.multiply(image, 2)
    image = tf.subtract(image, 1)
    # float64 to float32
    image = tf.cast(image, tf.float32)
    # HW to HWC
    return tf.expand_dims(image, -1)

def parse(example):
    ct, mr = ioutil.decode_example(example)

    return process(ct), process(mr)

def train(testdir, traindir, resultdir):
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    handle = tf.placeholder(tf.string, shape=[])

    test_records = [os.path.join(testdir, f) for f in os.listdir(testdir)]
    train_records = [os.path.join(traindir, f) for f in os.listdir(traindir)]

    test_dataset = tf.contrib.data.TFRecordDataset(test_records).map(parse).batch(4)
    train_dataset = tf.contrib.data.TFRecordDataset(train_records).map(parse).batch(4)

    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
        train_dataset.output_types, train_dataset.output_shapes)

    inputs, targets = iterator.get_next()

    test_iterator = test_dataset.make_one_shot_iterator()
    train_iterator = train_dataset.make_one_shot_iterator()

    test_handle_op = test_iterator.string_handle()
    train_handle_op = train_iterator.string_handle()

    outputs = model.unet(inputs)

    loss = tf.losses.mean_squared_error(outputs, targets)
    train = tf.train.GradientDescentOptimizer(1e-7).minimize(loss)

    with tf.train.MonitoredTrainingSession() as sess:
        test_handle, train_handle = sess.run([
            test_handle_op, train_handle_op])

        while not sess.should_stop():
            print('loss', sess.run(loss, feed_dict={handle: train_handle}))

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
