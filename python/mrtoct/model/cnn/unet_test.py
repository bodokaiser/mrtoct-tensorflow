import tensorflow as tf

from mrtoct import model


class UNetTest(tf.test.TestCase):

  def test_network_fn(self):
    x = tf.ones([10, 128, 128, 1])
    y = model.unet.network_fn(x)

    self.assertAllEqual(tf.TensorShape([10, 128, 128, 1]), y.shape)


if __name__ == '__main__':
  tf.test.main()
