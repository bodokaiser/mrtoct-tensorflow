import argparse
import tensorflow as tf

from mrtoct import data, ioutil, patch as p, util


def rebuild(input_path, output_path, params):
  tf_record_iterator = tf.python_io.tf_record_iterator(
      input_path, ioutil.TFRecordOptions)

  tf_record = tf.placeholder(tf.string)

  volume_transform = data.transform.Compose([
      data.transform.DecodeExample(),
      data.transform.Normalize(),
      data.transform.CenterMean(),
  ])

  volume = volume_transform(tf_record)
  vshape = tf.shape(volume)
  offset = tf.convert_to_tensor([32, 32, 32])

  indices = p.sample_meshgrid_3d(offset, vshape[:3] - offset, 10)
  indices_len = tf.shape(indices)[0]

  extract = data.transform.ExtractPatch([32, 32, 32, 1])

  def cond(i, *args):
    return i < indices_len

  def body(i, values, weights):
    start = indices[i] - offset // 2
    stop = start + offset

    patch = extract(indices[i], volume)
    index = util.meshgrid_3d(start, stop)

    update1 = tf.to_float(tf.scatter_nd(index, patch, vshape))
    update2 = tf.to_float(tf.scatter_nd(
        index, tf.to_float(patch > 0), vshape))

    return i + 1, values + update1, weights + update2

  _, values, weights = tf.while_loop(cond, body, [0, tf.zeros_like(volume, tf.float32),
                                                  tf.zeros_like(volume, tf.float32)], back_prop=False)

  cond = tf.not_equal(weights, 0)
  ones = tf.ones_like(weights)
  average = values / tf.where(cond, weights, ones)
  average32 = tf.image.convert_image_dtype(
      data.transform.Normalize()(average), tf.int32)

  with tf.Session() as sess:
    vol = sess.run(average32, feed_dict={tf_record: next(tf_record_iterator)})

    with tf.python_io.TFRecordWriter(output_path, ioutil.TFRecordOptions) as writer:
      encoder = ioutil.TFRecordEncoder()

      writer.write(encoder.encode(vol))


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  rebuild(args.input_path, args.output_path, None)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('generate')
  parser.add_argument('--input-path', required=True)
  parser.add_argument('--output-path', required=True)

  main(parser.parse_args())
