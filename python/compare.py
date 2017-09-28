import argparse
import tensorflow as tf

from mrtoct import data, ioutil


def compare(input_labels, input_filenames, output_filename):
  """Writes HTML file with per slice images of input volumes.

  Args:
    input_labels: labels of input, i.e. "Ground Truth"
    input_filenames: filenames to TFRecord volumes
    output_filename: filename of HTML output file
  """
  options = ioutil.TFRecordOptions.get_compression_type_string(
      ioutil.TFRecordOptions)

  label_dataset = (tf.contrib.data
                   .Dataset.from_tensor_slices(input_labels))

  volume_transform = data.transforms.Compose([
      data.transforms.DecodeExample(),
      data.transforms.CastType(),
      data.transforms.Normalize(),
  ])

  dataset = []

  for f in input_filenames:
    volume_dataset = (tf.contrib.data.TFRecordDataset(f, options)
        .map(volume_transform))

    tf.contrib.data.Dataset
      .zip(()) for f in input_filenames)

  tf.contrib.data.Dataset.zip((label_dataset, volume_dataset))

  tf.logging.info(f'converted {source} to {target}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  if len(args.inputs_labels) != len(args.input_filenames):
    raise argparse.ArgumentError('input labels and filenames differ in length')

  compare(args.inputs_labels, args.input_filenames, args.output_filename)


if __name__ == '__main__':
  parser=argparse.ArgumentParser('compare')
  parser.add_argument('--input-labels', required = True, nargs = '+')
  parser.add_argument('--input-filenames', required = True, nargs = '+')
  parser.add_argument('--output-filename', default = 'output.html')

  main(parser.parse_args())
