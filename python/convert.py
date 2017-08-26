import argparse
import os

import nibabel as nb
import tensorflow as tf

import ioutil

def subdirs(path):
    return list(filter(lambda f: os.path.isdir(os.path.join(path, f)),
        os.listdir(path)))

def main(args):
    for p in subdirs(args.inputdir):
        filename = os.path.join(args.outputdir, f'{p}.tfrecord')

        ctname = os.path.join(args.inputdir, p, 'ct.nii')
        ctvol = nb.load(ctname).get_data()

        mrname = os.path.join(args.inputdir, p, 'mr.nii')
        mrvol = nb.load(mrname).get_data()

        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in range(ctvol.shape[-1]):
                writer.write(ioutil.encode_example(ctvol[:,:,i], mrvol[:,:,i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', default='../data/nii')
    parser.add_argument('--outputdir', default='../data/tfrecord')

    main(parser.parse_args())
