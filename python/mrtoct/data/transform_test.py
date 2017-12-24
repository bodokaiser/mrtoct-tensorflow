import numpy as np
import tensorflow as tf

from mrtoct import data


class TransformTest(tf.test.TestCase):

  def test_compose(self):
    def fn1(x, y):
      return x + y

    def fn2(x):
      return 2 * x

    t = data.transform.Compose([fn1, fn2])

    self.assertEqual(8, t(3, 1))

  def test_expand_dims(self):
    x = tf.ones([3, 3])
    y = tf.ones([1, 3, 3])
    z = tf.ones([1, 3, 3, 1])

    t1 = data.transform.ExpandDims(0)
    t2 = data.transform.ExpandDims(-1)

    with self.test_session():
      self.assertAllEqual(y.eval(), t1(x).eval())
      self.assertAllEqual(z.eval(), t2(y).eval())

  def test_normalize(self):
    x = tf.constant([5, 11, 6, 1])
    y = tf.constant([0.4, 1.0, 0.5, 0.0])

    t = data.transform.Normalize()

    with self.test_session():
      self.assertAllClose(y.eval(), t(x).eval())

  def test_center_mean(self):
    x = tf.constant([0.4, 1.0, 0.5, 0.0])
    y = tf.constant([-0.2, 1.0, 0.0, -1.0])

    t = data.transform.CenterMean()

    with self.test_session():
      self.assertAllClose(y.eval(), t(x).eval())

  def test_uncenter_mean(self):
    x = tf.constant([-0.2, 1.0, -0.6, -1.0])
    y = tf.constant([0.4, 1.0, 0.2, 0.0])

    t = data.transform.UncenterMean()

    with self.test_session():
      self.assertAllClose(y.eval(), t(x).eval())

  def test_center_pad_3d(self):
    x = np.random.randn(100, 200, 300, 1)
    y = np.zeros([300, 300, 300, 1])
    z = np.zeros([400, 400, 400, 1])

    y[100:200, 50:250] = x
    z[150:250, 100:300, 50:350] = x

    t1 = data.transform.CenterPad3D(300, 300, 300)
    t2 = data.transform.CenterPad3D(400, 400, 400)

    with self.test_session():
      x = tf.constant(x)

      self.assertAllEqual(y, t1(x).eval())
      self.assertAllEqual(z, t2(x).eval())

  def test_index_crop_3d(self):
    x = np.random.randint(0, 256, size=(10, 10, 10, 1))
    y = x[4:7, 4:7, 4:7]
    z = x[4:6, 4:6, 4:6]

    t1 = data.transform.IndexCrop3D([3, 3, 3], [5, 5, 5])
    t2 = data.transform.IndexCrop3D([2, 2, 2], [5, 5, 5])

    with self.test_session():
      a = t1(x).eval()
      b = t2(x).eval()

      self.assertAllEqual(y, a)
      self.assertAllEqual(z, b)
      self.assertAllEqual(a[0:2, 0:2, 0:2], b)


if __name__ == '__main__':
  tf.test.main()
