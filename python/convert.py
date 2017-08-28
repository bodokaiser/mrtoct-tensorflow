import argparse
import os

import tensorflow as tf

from mrtoct import ioutil

def main(args):
    for e in os.listdir(args.input_path):
        source = os.path.join(args.input_path, e)

        if not os.path.isdir(source):
            continue

        for m in ['ct', 'mr']:
            target = os.path.join(args.output_path, f'{e}{m}.tfrecord')

            with tf.python_io.TFRecordWriter(target) as writer:
                volume = ioutil.read_nifti(os.path.join(source, f'{m}.nii'))

                for i in range(volume.shape[-1]):
                    writer.write(ioutil.encode_example(volume[:,:,i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='../data/nii')
    parser.add_argument('--output-path', default='../data/tfrecord')

    main(parser.parse_args())