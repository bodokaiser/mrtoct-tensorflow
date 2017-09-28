import argparse
import os
import tensorflow as tf

from mrtoct import ioutil


def convert(input_path, output_path):
  """Converts NIfTI volumes in `input_patch` to TFRecord at `output_path`.

  Args:
    input_path: path to directory with NIfTI files inside
    output_path: path to directory to write TFRecords to
  """
  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions

  os.makedirs(output_path, exist_ok=True)

  for fn, ext in map(os.path.splitext, os.listdir(input_path)):
    if ext != '.nii':
      continue

    source = os.path.join(input_path, f'{fn}.nii')
    target = os.path.join(output_path, f'{fn}.tfrecord')
    volume = ioutil.read_nifti(os.path.join(source))
    volume = ioutil.voxel_to_tensor_space(volume)

    with tf.python_io.TFRecordWriter(target, options) as writer:
      writer.write(encoder.encode(volume))

      tf.logging.info(f'converted {source} to {target}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  convert(args.input_path, args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('convert', description='''
    reads NIfTI volumes from input path and writes TFRecord to output path
  ''')
  parser.add_argument('--input-path', default='../data/nifti')
  parser.add_argument('--output-path', default='../data/tfrecord')

  main(parser.parse_args())
