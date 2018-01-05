import argparse
import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct import ioutil, data


def compare(volume, slice, inputs_path, outputs_path, targets_path):
  pre_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.ConstNormalization(tf.uint16.max),
  ])
  post_transform = data.transform.CropOrPad2D(384, 384)

  inputs_dataset = (tf.data
                    .TFRecordDataset(inputs_path, ioutil.TFRecordCString)
                    .map(pre_transform)
                    .skip(volume).take(1)
                    .apply(tf.contrib.data.unbatch())
                    .map(post_transform)
                    .skip(slice))
  outputs_dataset = (tf.data
                     .TFRecordDataset(outputs_path, ioutil.TFRecordCString)
                     .map(pre_transform)
                     .skip(volume).take(1)
                     .apply(tf.contrib.data.unbatch())
                     .map(post_transform)
                     .skip(slice))
  targets_dataset = (tf.data
                     .TFRecordDataset(targets_path, ioutil.TFRecordCString)
                     .map(pre_transform)
                     .skip(volume).take(1)
                     .apply(tf.contrib.data.unbatch())
                     .map(post_transform)
                     .skip(slice))

  dataset = (tf.data
             .Dataset.zip((inputs_dataset, outputs_dataset, targets_dataset)))

  volumes = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as session:
    mr, re, ct = session.run(volumes)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(mr, cmap='gray')
    axes[1].imshow(re, cmap='gray')
    axes[2].imshow(ct, cmap='gray')
    plt.show()


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  compare(volume=args.volume,
          slice=args.slice,
          inputs_path=args.inputs_path,
          outputs_path=args.outputs_path,
          targets_path=args.targets_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('compare')
  parser.add_argument('--slice', type=int, required=True)
  parser.add_argument('--volume', type=int, required=True)
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--targets-path', required=True)

  main(parser.parse_args())
