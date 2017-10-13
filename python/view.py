import argparse
import nibabel
import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct import data, ioutil


def view(input_path):
  options = ioutil.TFRecordOptions
  cstring = options.get_compression_type_string(options)

  transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.CastType(),
      data.transform.Normalize(),
  ])

  dataset = tf.data.TFRecordDataset(input_path, cstring).map(transform)

  volume = dataset.make_one_shot_iterator().get_next()

  with tf.train.MonitoredTrainingSession() as session:
    while not session.should_stop():
      v = session.run(volume)

      fig, axes = plt.subplots(2, 1)
      axes[0].imshow(v[100, :, :, 0])
      axes[1].imshow(v[100, :, :, 1])
      plt.show()


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  view(args.input_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('view')
  parser.add_argument('--input-path', required=True)

  main(parser.parse_args())
