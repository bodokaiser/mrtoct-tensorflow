import tensorflow as tf

from mrtoct import model


class UNetTest(tf.test.TestCase):

  def setUp(self):
    self.params = tf.contrib.training.HParams(num_filters=64)

  def test_unet(self):
    network = model.unet.generator_network(self.params)

    x = tf.zeros([10, 64, 64, 1])
    y = network(x)

    self.assertAllEqual(x.shape, y.shape)


if __name__ == '__main__':
  tf.test.main()
