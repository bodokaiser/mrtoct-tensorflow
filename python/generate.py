import argparse
import tensorflow as tf

from mrtoct import data, ioutil, patch as p, util, model


def generate(input_path, output_path, chkpt_path, params):
  """Generates CT from MR volumes.

  Args:
    input_path: path to tfrecord with MR volumes
    output_path: path where to write tfrecord with CT voluems to
    chkpt_path: path to trained checkpoint files
    params: hyper params for model
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions
  compstr = options.get_compression_type_string(options)

  volume_shape = [260, 340, 360, 1]

  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
      data.transform.CenterPad(volume_shape),
      data.transform.CenterMean(),
      data.transform.CastType(tf.float32),
  ])
  volume_dataset = data.TFRecordDataset(
      input_path, compstr).map(volume_transform)

  volume = volume_dataset.make_one_shot_iterator().get_next()

  vshape = tf.convert_to_tensor(volume_shape)
  pshape = tf.convert_to_tensor(params.patch_shape)

  indices = p.sample_meshgrid_3d(
      pshape[:3], vshape[:3] - pshape[:3], params.sample_delta)
  indices_len = tf.shape(indices)[0]

  patch_transform = data.transform.Compose([
      data.transform.ExtractPatch(pshape),
      data.transform.ExpandDims(0),
  ])

  def cond(i, *args):
    return i < indices_len

  def body(i, values, weights):
    start = indices[i] - pshape[:3] // 4
    stop = start + pshape[:3] // 2

    patch_in = tf.reshape(patch_transform(indices[i], volume),
                          [1, 32, 32, 32, 1])
    with tf.variable_scope('generator'):
      patch_out = model.gan.synthesis.generator(patch_in, params)[0]
    index = util.meshgrid_3d(start, stop)

    update1 = tf.to_float(tf.scatter_nd(index, patch_out, vshape))
    update2 = tf.to_float(tf.scatter_nd(
        index, tf.to_float(patch_out > -1), vshape))

    values += update1
    weights += update2

    values.set_shape(volume_shape)
    weights.set_shape(volume_shape)

    return i + 1, values, weights

  _, values, weights = tf.while_loop(
      cond, body, [0,
                   tf.zeros(volume_shape[:3], tf.float32),
                   tf.zeros(volume_shape[:3], tf.float32)],
      back_prop=False)

  final_transform = data.transform.Compose([
      data.transform.UncenterMean(),
      data.transform.Normalize(),
      lambda x: tf.image.convert_image_dtype(x, tf.int32),
  ])

  cond = tf.not_equal(weights, 0)
  ones = tf.ones_like(weights)
  average = final_transform(values / tf.where(cond, weights, ones))

  reader = tf.train.NewCheckpointReader(chkpt_path)
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
  writer = tf.python_io.TFRecordWriter(output_path, options)

  tf.logging.info('Computation graph completed')

  with tf.Session(config=config) as sess:
    saver.restore(sess, chkpt_path)

    try:
      while True:
        writer.write(encoder.encode(sess.run(average)))

        tf.logging.info('Iteration completed')
    except tf.errors.OutOfRangeError:
      pass
    finally:
      writer.flush()
      writer.close()

      tf.logging.info('Writer closed')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=5,
      patch_shape=[32, 32, 32, 1])
  hparams.parse(args.hparams)

  generate(args.input_path, args.output_path, args.chkpt_path, hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('generate')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--chkpt-path', required=True)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
