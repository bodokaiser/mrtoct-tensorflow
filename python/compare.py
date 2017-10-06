import argparse
import tensorflow as tf

from mrtoct import data, ioutil, html


def compare(inputs, target, labels, steps):
  """Writes HTML file with per slice images of input volumes.

  Args:
    labels: labels of input (i.e. "Ground Truth") to use as table head
    inputs: filenames to TFRecord volumes
    target: filename of HTML output file
    steps: distance between slices
  """
  options = ioutil.TFRecordOptions
  cstring = options.get_compression_type_string(options)

  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.CastType(),
      data.transform.Normalize(),
  ])
  slice_transform = data.transform.ExtractSlice()

  def slice_dataset(filename):
    return (tf.data.Dataset
            .zip((tf.data.Dataset.range(0, 100, steps),
                  tf.data.TFRecordDataset(filename, cstring)
                  .map(volume_transform)
                  .cache().repeat()))
            .map(slice_transform))

  slices = (tf.data.Dataset
            .zip(tuple(slice_dataset(f) for f in inputs))
            .make_one_shot_iterator().get_next())

  with tf.train.MonitoredTrainingSession() as sess:
    step = 0

    with html.ImageWriter(target) as writer:
      if len(labels) > 0:
        writer.write_head(labels)

      while not sess.should_stop():
        writer.write_row(sess.run(slices))

        step += 1

  tf.logging.info(f'wrote {step} slices to {target}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  if args.labels and len(args.labels) != len(args.input_filenames):
    raise ValueError('input labels and filenames differ in length')

  compare(args.input_filenames, args.output_filename, args.labels, args.steps)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('compare')
  parser.add_argument('--steps', type=int, default=4)
  parser.add_argument('--labels', nargs='+', default=[])
  parser.add_argument('--input-filenames', required=True, nargs='+')
  parser.add_argument('--output-filename', default='output.html')

  main(parser.parse_args())
