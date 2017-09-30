import argparse
import os
import tensorflow as tf

from mrtoct import ioutil


def convert(input_paths, output_path):
  """Converts volumes from NIfTI to TFRecord format.

  Args:
    input_path: list of paths to nifti volumes
    output_path: path to write TFRecords to
  """
  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions

  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  with tf.python_io.TFRecordWriter(output_path, options) as writer:
    for input_path in input_paths:
      volume = ioutil.read_nifti(input_path)
      volume = ioutil.voxel_to_tensor_space(volume)

      writer.write(encoder.encode(volume))

      tf.logging.info(f'Wrote {input_path} to {output_path}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  convert(args.input_paths, args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('convert')
  parser.add_argument('--input-paths', required=True, nargs='+')
  parser.add_argument('--output-path', required=True)

  main(parser.parse_args())
