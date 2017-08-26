import argparse
import os

import tensorflow as tf

from matplotlib import pyplot as plt

import ioutil

TEST = ['p001', 'p002', 'p003', 'p004']
TRAIN = ['p005', 'p006', 'p007', 'p101', 'p102', 'p103', 'p104', 'p105',
         'p106', 'p107', 'p108', 'p109', 't001']

def parse(example):
    return ioutil.decode_example(example)

def main(args):
    handle = tf.placeholder(tf.string, shape=[])

    test_dataset = tf.contrib.data.TFRecordDataset([
        os.path.join(args.inputdir, f'{s}.tfrecord') for s in TEST]).map(parse)
    train_dataset = tf.contrib.data.TFRecordDataset([
        os.path.join(args.inputdir, f'{s}.tfrecord') for s in TRAIN]).map(parse)

    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
        train_dataset.output_types, train_dataset.output_shapes)

    next_element = iterator.get_next()

    test_iterator = test_dataset.make_one_shot_iterator()
    train_iterator = train_dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        for i in range(20):
            test_handle = sess.run(test_iterator.string_handle())
            train_handle = sess.run(train_iterator.string_handle())

            res = sess.run(next_element, feed_dict={handle: train_handle})

            fig, axes = plt.subplots(2)
            axes[0].imshow(res[0])
            axes[1].imshow(res[1])
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default='../data/tfrecord')
    parser.add_argument('--outputdir', default='results')

    main(parser.parse_args())
