import tensorflow as tf

from mrtoct import model


class SynthesisTest(tf.test.TestCase):

  def setUp(self):
    self.params = tf.contrib.training.HParams()

  def test_generator(self):
    network = model.synthesis.generator_network(self.params)

    x = tf.ones([10, 32, 32, 32, 1])
    y = network(x)

    self.assertAllEqual(x.shape, y.shape)

  def test_discriminator(self):
    network = model.synthesis.discriminator_network(self.params)

    x = tf.ones([10, 32, 32, 32, 1])
    network(x)


if __name__ == '__main__':
  tf.test.main()
