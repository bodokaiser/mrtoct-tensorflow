import numpy as np
import tensorflow as tf

from mrtoct import ioutil


class TFRecordTest(tf.test.TestCase):

  def setUp(self):
    self.encoder = ioutil.TFRecordEncoder()
    self.decoder = ioutil.TFRecordDecoder()

  def test_encoder_decoder(self):
    volume = np.random.randint(0, 1000, [30, 50, 30])

    tfrecord = self.encoder.encode(volume)

    with self.test_session():
      self.assertAllEqual(volume, self.decoder.decode(tfrecord).eval())


if __name__ == '__main__':
  tf.test.main()
