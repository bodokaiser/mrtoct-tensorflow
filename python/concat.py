import argparse
import tensorflow as tf

from mrtoct import ioutil, data


def concat(input_paths, output_path):
  """Concats TFRecord volumes along channel axis.

  Args:
    input_path: list of paths to nifti volumes
    output_path: path to write TFRecords to
  """
  encoder = ioutil.TFRecordEncoder()
  options = ioutil.TFRecordOptions

  ctype = options.get_compression_type_string(options)

  transform = data.transform.DecodeExample()

  datasets = [data.TFRecordDataset(f, ctype).map(transform)
              for f in input_paths]
  dataset = data.Dataset.zip(tuple(datasets))

  volumes = dataset.make_one_shot_iterator().get_next()
  volume = tf.concat(volumes, -1)

  writer = tf.python_io.TFRecordWriter(output_path, options)

  with tf.Session() as session:
    i = 0
    try:
      while True:
        writer.write(encoder.encode(session.run(volume)))

        tf.logging.info(f'Concated {i}-th volumes to {output_path}')

        i += 1
    except tf.errors.OutOfRangeError:
      tf.logging.info('Completed iterations')
    except Exception as e:
      tf.logging.error(f'Error occured {e}')
    finally:
      writer.close()

      tf.logging.info(f'Closed {output_path}')


def main(args):
  tf.logging.set_verbosity(tf.logging.INFO)

  concat(args.input_paths, args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('concat')
  parser.add_argument('--input-paths', required=True, nargs='+')
  parser.add_argument('--output-path', required=True)

  main(parser.parse_args())
