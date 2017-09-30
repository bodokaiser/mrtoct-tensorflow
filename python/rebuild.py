import argparse
import tensorflow as tf

from mrtoct import ioutil, data, patch, util


def rebuild(input_path, output_path, params, batch_size):
  """Rebuilds TFRecord from input_path by applying model on patches.

  Args:
    input_path: path to read tfrecord from
    output_path: path to write tfrecord to
    params: hyper parameters to use for model
    stack: if `True` stack new build to channels
  """
  options = ioutil.TFRecordOptions

  for i, tfrecord in enumerate(tf.python_io.tf_record_iterator(
          input_path, options)):
    tf.reset_default_graph()

    with tf.name_scope('config'):
      pshape = tf.convert_to_tensor(params.patch_shape)
      vshape = tf.convert_to_tensor(params.volume_shape)

    with tf.name_scope('volume'):
      volume_transform = data.transform.Compose([
          data.transform.DecodeExample(),
          data.transform.CastType(),
          data.transform.Normalize(),
          data.transform.CenterMean(),
          data.transform.CenterPad(vshape),
      ])

      volume_dataset = (data.Dataset
                        .from_tensors(tfrecord)
                        .map(volume_transform)
                        .cache())

      with tf.name_scope('index'):
        start = pshape[:3] // 2
        stop = vshape[:3] - start

        indices = patch.sample_meshgrid_3d(start, stop, params.sample_delta)
        indices_len = tf.to_int64(tf.shape(indices)[0])

        index_dataset = data.Dataset.from_tensor_slices(indices)

      with tf.name_scope('patch'):
        patch_transform = data.transform.ExtractPatch(pshape)

        patch_dataset = (data.Dataset
                         .zip((index_dataset,
                               volume_dataset.repeat(indices_len)))
                         .map(patch_transform))

      with tf.name_scope('combined'):
        def expand_index(index):
          offset = pshape[:3] // 2
          return util.meshgrid_3d(index - offset, index + offset, 1)

        combined_dataset = data.Dataset.zip(
            (index_dataset.map(expand_index), patch_dataset)).batch(batch_size)

    with tf.name_scope('iterator'):
      combined_iterator = combined_dataset.make_initializable_iterator()

      indices, patches = combined_iterator.get_next()

    with tf.name_scope('aggregation'):
      sma = patch.SparseMovingAverage(params.volume_shape[:3], name='build')

      update = sma.update(indices, patches[:, :, :, :, 0])
      average = sma.average()

    with tf.name_scope('session'):
      step = tf.assign_add(tf.train.get_or_create_global_step(), 1)

      scaffold = tf.train.Scaffold(init_op=tf.group(
          tf.global_variables_initializer(),
          combined_iterator.initializer))

    with tf.train.MonitoredTrainingSession(scaffold=scaffold) as sess:
      while not sess.should_stop():
        s, _, result = sess.run([step, update, average])

        if s % 100 == 0:
          tf.logging.info(f'Processing volume {i} with batch {s}')

    tf.logging.info(f'Completed rebuild of volume {i}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=8,
      patch_shape=[32, 32, 32, 1],
      volume_shape=[240, 320, 340, 1])
  hparams.parse(args.hparams)

  rebuild(args.input_path, args.output_path, hparams, args.batch_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('convert')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--batch-size', default=10)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
