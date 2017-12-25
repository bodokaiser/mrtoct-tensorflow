import argparse
import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct import ioutil, data


def compare(inputs_path, outputs_path, targets_path):
  transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
  ])

  inputs_dataset = (tf.data
                    .TFRecordDataset(inputs_path, ioutil.TFRecordCString)
                    .map(transform))
  outputs_dataset = (tf.data
                     .TFRecordDataset(outputs_path, ioutil.TFRecordCString)
                     .map(transform))
  targets_dataset = (tf.data
                     .TFRecordDataset(targets_path, ioutil.TFRecordCString)
                     .map(transform))

  dataset = (tf.data
             .Dataset.zip((inputs_dataset, outputs_dataset, targets_dataset)))

  volumes = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as session:
    mr, re, ct = session.run(volumes)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(mr[10, :, :, 0])
    axes[1].imshow(re[10, :, :, 0])
    axes[2].imshow(ct[10, :, :, 0])
    plt.show()


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  compare(inputs_path=args.inputs_path,
          outputs_path=args.outputs_path,
          targets_path=args.targets_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('compare')
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--targets-path', required=True)

  main(parser.parse_args())
