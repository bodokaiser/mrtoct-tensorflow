import tensorflow as tf

from mrtoct import model


class PixToPixGANTest(tf.test.TestCase):

  def test_generator_fn(self):
    x = tf.ones([10, 128, 128, 2])
    y = model.synthesis.generator_fn(x)

    self.assertAllEqual(x.shape, y.shape)

  def test_discriminator_fn(self):
    x = tf.ones([10, 128, 128, 1])
    y = tf.ones([10, 128, 128, 1])

    model.gan.pixtopix.discriminator_fn(x, y)


if __name__ == '__main__':
  tf.test.main()
