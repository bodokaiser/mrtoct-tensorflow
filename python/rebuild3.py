import argparse
import tensorflow as tf

from mrtoct import ioutil, data, model, patch as p


def rebuild(input_path, chkpt_path, output_path, params, batch_size):
  tfrecord = tf.placeholder(tf.string, name='tfrecord')

  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.CastType(),
      data.transform.Normalize(),
      data.transform.CenterMean(),
  ])

  volume = volume_transform(tfrecord)
  vshape = tf.shape(volume)
  pshape = tf.convert_to_tensor(params.patch_shape)

  off = pshape[:3] // 2
  size = vshape[:3] - off

  indices = p.sample_meshgrid_3d(off, size, params.sample_delta)
  indices_len = tf.to_int64(tf.shape(indices)[0])

  index_dataset = data.Dataset.from_tensor_slices(indices)
  volume_dataset = data.Dataset.from_tensors(volume)

  patch_transform = data.transform.ExtractPatch(pshape)

  patch_dataset = (data.Dataset
                   .zip((index_dataset, volume_dataset))
                   .map(patch_transform))

  patch_iterator = patch_dataset.make_initializable_iterator()
  patch = patch_iterator.get_next()

  with tf.Session() as session:
    encoder = ioutil.TFRecordEncoder()
    options = ioutil.TFRecordOptions

    for r in tf.python_io.tf_record_iterator(input_path, options):
      try:
        session.run(patch_iterator.initializer,
                    feed_dict={tfrecord: r})
        while True:
          print(session.run(patch, feed_dict={tfrecord: r}).shape)
      except tf.errors.OutOfRangeError:
        pass


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=10,
      patch_shape=[32, 32, 32, 1],
      generator=model.gan.synthesis.generator_network)
  hparams.parse(args.hparams)

  rebuild(args.input_path,
          args.chkpt_path,
          args.output_path,
          hparams,
          args.batch_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('patch')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--chkpt-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--batch-size', default=40)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
