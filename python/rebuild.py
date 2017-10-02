import argparse
import tensorflow as tf

from mrtoct import ioutil, data, model, patch, util


def rebuild(input_path, chkpt_path, output_path, params, batch_size):
  """Rebuilds TFRecord from input_path by applying model on patches.

  Args:
    input_path: path to read tfrecord from
    input_path: path to read variables from
    output_path: path to write tfrecord to
    params: hyper parameters to use for model
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.device('/cpu:0'):
    with tf.name_scope('shapes'):
      vshape = tf.convert_to_tensor(params.volume_shape, name='volume_shape')
      pshape = tf.convert_to_tensor(params.patch_shape, name='patch_shape')

    with tf.name_scope('indices'):
      off = pshape[:3] // 2
      size = vshape[:3] - off

      indices = patch.sample_meshgrid_3d(off, size, params.sample_delta)
      indices_len = tf.to_int64(tf.shape(indices)[0])

    with tf.name_scope('dataset'):
      options = ioutil.TFRecordOptions.get_compression_type_string(
          ioutil.TFRecordOptions)

      volume_transform = data.transform.Compose([
          data.transform.DecodeExample(),
          data.transform.CastType(),
          data.transform.Normalize(),
          data.transform.CenterMean(),
          data.transform.CenterPad(vshape),
      ])

      with tf.name_scope('index'):
        index_dataset = data.Dataset.from_tensor_slices(indices)

      with tf.name_scope('volume'):
        volume_dataset = data.TFRecordDataset(
            input_path, options).map(volume_transform).cache()

        patch_transform = data.transform.ExtractPatch(pshape)

      def extract_patches(volume):
        volume_dataset = (data.Dataset.from_tensors(volume)
                          .repeat(indices_len))

        return (data.Dataset
                .zip((index_dataset, volume_dataset))
                .map(patch_transform))

        return extract_patches

      def expand_index(index):
        offset = pshape[:3] // 4

        return util.meshgrid_3d(index - offset, index + offset, 1)

      with tf.name_scope('patch'):
        patch_dataset = (data.Dataset
                         .zip((index_dataset.map(expand_index),
                               volume_dataset.flat_map(extract_patches)))
                         .enumerate().batch(1))

      with tf.name_scope('iterator'):
        patch_iterator = patch_dataset.make_initializable_iterator()

        p_step, p_indices, p_values = patch_iterator.get_next()

    with tf.device('/gpu:0'):
      with tf.name_scope('model'):
        with tf.variable_scope('generator'):
          outputs = params.generator(params)(p_values)
          outputs = data.transform.UncenterMean()(outputs[:, :, :, :, 0])

    with tf.device('/cpu:0'):
      with tf.name_scope('aggregator'):
        sma = patch.SparseMovingAverage(params.volume_shape[:3], name='build')

        update = sma.update(p_indices, outputs)
        resets = sma.initializer()
        average = sma.average()
        average32 = tf.image.convert_image_dtype(average, tf.int32)

    def optimistic_saver(save_file):
      reader = tf.train.NewCheckpointReader(save_file)
      saved_shapes = reader.get_variable_to_shape_map()
      var_names = sorted([(var.name, var.name.split(':')[0])
                          for var in tf.global_variables()
                          if var.name.split(':')[0] in saved_shapes])
      restore_vars = []
      name2var = dict(zip(map(lambda x: x.name.split(
          ':')[0], tf.global_variables()), tf.global_variables()))
      with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
          curr_var = name2var[saved_var_name]
          var_shape = curr_var.get_shape().as_list()
          if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
      saver = tf.train.Saver(restore_vars)
      return saver

    tf.logging.info('Constructed computation graph')

    encoder = ioutil.TFRecordEncoder()
    options = ioutil.TFRecordOptions

    saver = optimistic_saver(chkpt_path)

    summary_writer = tf.summary.FileWriter('foo', tf.get_default_graph())

    tf.summary.histogram('average', average)
    tf.summary.histogram('patches', outputs)

    summary_op = tf.summary.merge_all()

    record_writer = tf.python_io.TFRecordWriter(output_path, options)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
      sess.run(init)
      saver.restore(sess, chkpt_path)

      num_batches_per_volume = sess.run(indices_len) // batch_size

      try:
        while True:
          s, _ = sess.run([p_step, update])

          if s % 100 == 0:
            summary = sess.run(summary_op)

            summary_writer.add_summary(summary, s)

            tf.logging.info(f'Processed step {s}')
          if s % num_batches_per_volume == 0:
            record_writer.write(encoder.encode(sess.run(average32)))
            sess.run(resets)

            tf.logging.info(f'Processed volume {s // num_batches_per_volume}')
      except tf.errors.OutOfRangeError:
        tf.logging.info('Iteration complete')
      finally:
        record_writer.close()

        tf.logging.info('Rebuilt complete')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=10,
      patch_shape=[32, 32, 32, 1],
      volume_shape=[240, 340, 360, 1],
      generator=model.gan.synthesis.generator_network)
  hparams.parse(args.hparams)

  rebuild(args.input_path, args.chkpt_path, args.output_path, hparams,
          args.batch_size)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('convert')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--chkpt-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--batch-size', default=40)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
