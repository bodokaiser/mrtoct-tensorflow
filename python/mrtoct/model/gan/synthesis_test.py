import tensorflow as tf

from mrtoct import model


class SynthesisGANTest(tf.test.TestCase):

  def setUp(self):
    self.params = tf.contrib.training.HParams()

  def test_generator_fn(self):
    x1 = tf.ones([10, 32, 32, 32, 1])
    x2 = tf.ones([10, 32, 32, 32, 2])

    with tf.variable_scope('generator1'):
      y1 = model.synthesis.generator_fn(x1)

    with tf.variable_scope('generator2'):
      y2 = model.synthesis.generator_fn(x2)

    self.assertAllEqual(tf.TensorShape([10, 16, 16, 16, 1]), y1.shape)
    self.assertAllEqual(tf.TensorShape([10, 16, 16, 16, 1]), y2.shape)

  def test_discriminator_fn(self):
    x = tf.ones([10, 32, 32, 32, 1])
    y = model.synthesis.discriminator_fn(x)


if __name__ == '__main__':
  tf.test.main()
