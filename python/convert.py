import argparse
import os

import nibabel as nb
import tensorflow as tf

from ioutil import tfrecord

def subdirs(path):
    return list(filter(lambda f: os.path.isdir(os.path.join(path, f)),
        os.listdir(path)))

def main(args):
    for p in subdirs(args.inputdir):
        filename = os.path.join(args.outputdir, f'{p}.tfrecord')

        ct = nb.load(os.path.join(args.inputdir, p, 'ct.nii')).get_data()
        mr = nb.load(os.path.join(args.inputdir, p, 'mr.nii')).get_data()

        with tf.python_io.TFRecordWriter(filename) as writer:
            assert ct.shape[-1] == mr.shape[-1], 'incompatible shapes'

            for i in range(ct.shape[-1]):
                writer.write(tfrecord.encode_example(ct[:,:,i], mr[:,:,i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default='../data/nii')
    parser.add_argument('--outputdir', default='../data/tfr')

    main(parser.parse_args())
