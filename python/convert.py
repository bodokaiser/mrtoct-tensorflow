import argparse
import os
import tensorflow as tf

from mrtoct import ioutil


def convert(input_path, output_path):
    options = ioutil.TFRecordOptions
    encoder = ioutil.TFRecordEncoder()

    os.makedirs(output_path, exist_ok=True)

    for fn, ext in map(os.path.splitext, os.listdir(input_path)):
        if ext != '.nii':
            continue

        source = os.path.join(input_path, f'{fn}.nii')
        target = os.path.join(output_path, f'{fn}.tfrecord')
        volume = ioutil.read_nifti(os.path.join(source))

        with tf.python_io.TFRecordWriter(target, options) as writer:
            writer.write(encoder.encode(volume))


def main(args):
    convert(args.input_path, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default='../data/nifti')
    parser.add_argument('--output-path', default='../data/tfrecord')

    main(parser.parse_args())
