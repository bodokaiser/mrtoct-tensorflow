import argparse
import os
import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct import data
from mrtoct.utils import count

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = (data.make_zipped_dataset('../data/tfrecord')
        .filter(data.filter_nans)
        .filter(data.filter_incomplete))

    iterator = dataset.make_one_shot_iterator()

    mr_op, ct_op = iterator.get_next()

    mr_sum_op = count(tf.greater(mr_op, tf.reduce_min(mr_op)))
    ct_sum_op = count(tf.greater(ct_op, tf.reduce_min(ct_op)))

    step = 0

    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            mr, ct, mr_sum, ct_sum = sess.run([mr_op, ct_op,
                mr_sum_op, ct_sum_op])

            if args.show:
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(mr[:,:,0])
                axes[1].imshow(ct[:,:,0])
                plt.show()
            step += 1

            tf.logging.info(f'step: {step}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--input-path', default='../data/tfrecord')

    main(parser.parse_args())