import tensorflow as tf

from mrtoct import model


class Pix2PixTest(tf.test.TestCase):

  def setUp(self):
    self.params = tf.contrib.training.HParams(num_filters=64)

  def test_generator(self):
    network = model.pix2pix.generator_network(self.params)

    x = tf.ones([10, 128, 128, 1])
    y = network(x)

    self.assertAllEqual(x.shape, y.shape)

  def test_discriminator(self):
    network = model.pix2pix.discriminator_network(self.params)

    x = tf.ones([10, 128, 128, 2])
    network(x)


if __name__ == '__main__':
  tf.test.main()
