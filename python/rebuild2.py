import argparse
import tensorflow as tf

from mrtoct import ioutil, data, model, patch as p


def rebuild(input_path, chkpt_path, output_path, params, batch_size):
  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions

  with tf.name_scope('shapes'):
    vshape = tf.convert_to_tensor(params.volume_shape, name='volume')
    pshape = tf.convert_to_tensor(params.patch_shape, name='patch')

  with tf.name_scope('volume'):
    gen = tf.python_io.tf_record_iterator(input_path, options)

    volume_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.CastType(),
        data.transform.Normalize(),
        data.transform.CenterMean(),
        data.transform.CenterPad(vshape),
    ])

    volume = volume_transform(next(gen))

  with tf.name_scope('indices'):
    off = pshape[:3] // 2
    size = vshape[:3] - off

    indices = p.sample_meshgrid_3d(off, size, params.sample_delta)
    indices_len = tf.to_int64(tf.shape(indices)[0])

  with tf.name_scope('dataset'):
    with tf.name_scope('index'):
      index_dataset = data.Dataset.from_tensor_slices(indices).shuffle(1000)

    with tf.name_scope('volume'):
      volume_dataset = data.Dataset.from_tensors(volume).cache()

    with tf.name_scope('patch'):
      patch_transform = data.transform.ExtractPatch(pshape)

      patch_dataset = (data.Dataset
                       .zip((index_dataset, volume_dataset))
                       .map(patch_transform))

  with tf.name_scope('iterator'):
    patch_iterator = patch_dataset.make_one_shot_iterator()

    patch = patch_iterator.get_next()

  with tf.Session() as sess:
    for i in range(100000):
      print(sess.run(patch))


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=10,
      patch_shape=[32, 32, 32, 1],
      volume_shape=[240, 340, 360, 1],
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
