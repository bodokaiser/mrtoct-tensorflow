import argparse
import tensorflow as tf

from mrtoct import data, ioutil, patch as p, util, model


def generate(input_path, output_path, chkpt_path, params):
  config = tf.ConfigProto(device_count={'GPU': 0})
  config.log_device_placement = True

  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions
  compstr = options.get_compression_type_string(options)

  volume_shape = [300, 380, 400, 1]
  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
      data.transform.CenterPad3D(*volume_shape[:3]),
      data.transform.Lambda(lambda x: tf.reshape(x, volume_shape)),
  ])
  volume_dataset = tf.data.TFRecordDataset(
      input_path, compstr).map(volume_transform)

  volume = volume_dataset.make_one_shot_iterator().get_next()

  vshape = tf.shape(volume)
  pshape = tf.convert_to_tensor(params.patch_shape)

  indices = p.sample_meshgrid_3d(
      pshape[:3], vshape[:3] - pshape[:3], params.sample_delta)
  indices_len = tf.shape(indices)[0]

  def extract_patch_at(index, volume):
    i = index[0] - 16
    j = index[1] - 16
    k = index[2] - 16

    return volume[i:i + 32, j:j + 32, k:k + 32]

  patch_transform = data.transform.Compose([
      extract_patch_at,
      data.transform.ExpandDims(0),
  ])

  def cond(i, *args):
    return i < indices_len

  def body(i, values, weights):
    start = indices[i] - pshape[:3] // 4
    stop = start + pshape[:3] // 2

    patch_in = tf.reshape(patch_transform(indices[i], volume),
                          [1, 32, 32, 32, 1])
    with tf.variable_scope('Generator'):
      patch_out = model.gan.synthesis.generator_fn(
          patch_in, 'channels_last')[0]
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
                   tf.zeros_like(volume, tf.float32),
                   tf.zeros_like(volume, tf.float32)], back_prop=False)

  final_transform = data.transform.Compose([
      data.transform.Normalize(),
      lambda x: tf.image.convert_image_dtype(x, tf.int32),
  ])

  cond = tf.not_equal(weights, 0)
  ones = tf.ones_like(weights)
  average = final_transform(values / tf.where(cond, weights, ones))

  saver = tf.train.Saver()
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
      sample_delta=3,
      patch_shape=[32, 32, 32, args.iteration])
  hparams.parse(args.hparams)

  generate(args.inputs_path, args.outputs_path, args.checkpoint_path, hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('predict')
  parser.add_argument('--iteration', type=int, required=True)
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--outputs-path', required=True)
  parser.add_argument('--checkpoint-path', required=True)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
