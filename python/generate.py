import argparse
import tensorflow as tf

from mrtoct import data, ioutil, patch as p, util


def generate(input_path, output_path, chkpt_path, params):
  """Generates CT from MR volumes.

  Args:
    input_path: path to tfrecord with MR volumes
    output_path: path where to write tfrecord with CT voluems to
    chkpt_path: path to trained checkpoint files
    params: hyper params for model
  """

  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions
  compstr = options.get_compression_type_string(options)

  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
      data.transform.CenterMean(),
  ])
  volume_dataset = data.TFRecordDataset(
      input_path, compstr).map(volume_transform)

  volume = volume_dataset.make_one_shot_iterator().get_next()

  vshape = tf.shape(volume)
  pshape = tf.convert_to_tensor(params.patch_shape)

  indices = p.sample_meshgrid_3d(
      pshape[:3], vshape[:3] - pshape[:3], params.sample_delta)
  indices_len = tf.shape(indices)[0]

  patch_transform = data.transform.ExtractPatch(pshape)

  def cond(i, *args):
    return i < indices_len

  def body(i, values, weights):
    start = indices[i] - pshape[:3] // 2
    stop = start + pshape[:3]

    patch = patch_transform(indices[i], volume)
    index = util.meshgrid_3d(start, stop)

    update1 = tf.to_float(tf.scatter_nd(index, patch, vshape))
    update2 = tf.to_float(tf.scatter_nd(
        index, tf.to_float(patch > -1), vshape))

    return i + 1, values + update1, weights + update2

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

  with tf.Session() as sess:
    vol = sess.run(average)

    with tf.python_io.TFRecordWriter(output_path, options) as writer:
      writer.write(encoder.encode(vol))


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  hparams = tf.contrib.training.HParams(
      sample_delta=20,
      patch_shape=[32, 32, 32, 1])
  hparams.parse(args.hparams)

  generate(args.input_path, args.output_path, hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('generate')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--hparams', type=str, default='')

  main(parser.parse_args())
