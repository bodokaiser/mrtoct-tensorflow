import argparse
import tensorflow as tf

from mrtoct import ioutil, data, patch as p


def patch(input_path, output_path, params):
  """Converts TFRecord volumes to patches.

  Args:
    input_path: path to tfrecord volume from which to create patches from
    output_path: path to tfrecord volume to which to write patches to
  """
  with tf.name_scope('config'):
    encoder = ioutil.TFRecordEncoder()
    options = ioutil.TFRecordOptions

    vshape = tf.convert_to_tensor(params.volume_shape, name='volume_shape')
    pshape = tf.convert_to_tensor(params.patch_shape, name='patch_shape')

  with tf.name_scope('indices'):
    off = tf.convert_to_tensor([16, 16, 16])
    size = vshape[:3] - off
    # if we use pshape here seed will not match
    #off = pshape[:3] // 2
    #size = vshape[:3] - off

    indices = p.sample_uniform_3d(off, size, params.sample_num, params.seeds)

  with tf.name_scope('dataset'):
    compression = ioutil.TFRecordOptions.get_compression_type_string(
        options)

    volume_transform = data.transform.Compose([
        data.transform.DecodeExample(),
        data.transform.CenterPad(vshape),
    ])
    patch_transform = data.transform.ExtractPatch(pshape)

    with tf.name_scope('index'):
      index_dataset = data.Dataset.from_tensor_slices(indices)

    with tf.name_scope('volume'):
      volume_dataset = data.TFRecordDataset(
          input_path, compression).map(volume_transform).cache()

    def extract_patches(volume):
      volume_dataset = (data.Dataset
                        .from_tensors(volume)
                        .repeat(params.sample_num))

      return (data.Dataset
              .zip((index_dataset, volume_dataset))
              .map(patch_transform))

    with tf.name_scope('patch'):
      patch_dataset = volume_dataset.flat_map(extract_patches)

  with tf.name_scope('iterator'):
    patch_iterator = patch_dataset.make_initializable_iterator()

    patches = patch_iterator.get_next()

  with tf.name_scope('session'):
    step = tf.assign_add(tf.train.get_or_create_global_step(), 1)

    scaffold = tf.train.Scaffold(init_op=tf.group(
        tf.global_variables_initializer(), patch_iterator.initializer))

  with tf.python_io.TFRecordWriter(output_path, options) as writer:
    with tf.train.MonitoredTrainingSession(scaffold=scaffold) as sess:
      while not sess.should_stop():
        s, vol = sess.run([step, patches])

        writer.write(encoder.encode(vol))

        if s % 1000 == 0:
          tf.logging.info(f'Wrote {s} patches to {output_path}')

  tf.logging.info(f'Completed patches.')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_num=1000,
      seeds=[args.seed,
             args.seed * 2,
             args.seed * 3],
      patch_shape=[args.patch_size,
                   args.patch_size,
                   args.patch_size, 1],
      volume_shape=[260, 340, 340, 1])
  hparams.parse(args.hparams)

  patch(args.input_path, args.output_path, hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('patch')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--patch-size', type=int, required=True)
  parser.add_argument('--seed', type=int, default=1000)
  parser.add_argument('--hparams', default='')

  main(parser.parse_args())
