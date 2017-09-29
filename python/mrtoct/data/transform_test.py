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

  def test_cast_type(self):
    x = tf.constant(1, tf.int32)
    y = tf.constant(1.0, tf.float32)

    t = data.transform.CastType()

    with self.test_session():
      self.assertEqual(y.eval(), t(x).eval())

  def test_expand_dims(self):
    x = tf.ones([3, 3])
    y = tf.ones([1, 3, 3])
    z = tf.ones([1, 3, 3, 1])

    t1 = data.transform.ExpandDims(0)
    t2 = data.transform.ExpandDims(-1)

    with self.test_session():
      self.assertAllEqual(y.eval(), t1(x).eval())
      self.assertAllEqual(z.eval(), t2(y).eval())

  def test_center_pad(self):
    x = np.random.randn(100, 200, 300, 1)
    y = np.zeros([300, 300, 300, 1])
    z = np.zeros([400, 400, 400, 1])

    y[100:200, 50:250, :, :] = x
    z[150:250, 100:300, 50:350, :] = x

    t1 = data.transform.CenterPad([300, 300, 300, 1])
    t2 = data.transform.CenterPad([400, 400, 400, 1])

    with self.test_session():
      x = tf.constant(x)

      self.assertAllEqual(y, t1(x).eval())
      self.assertAllEqual(z, t2(x).eval())

  def test_center_crop(self):
    x = np.random.randn(100, 200, 300, 1)

    t1 = data.transform.CenterCrop([100, 100, 100, 1])
    t2 = data.transform.CenterCrop([50, 50, 50, 1])

    with self.test_session():
      y = tf.constant(x)

      self.assertAllEqual(x[:, 50:150, 100:200], t1(y).eval())
      self.assertAllEqual(x[25:75, 75:125, 125:175], t2(y).eval())

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

  def test_extract_slice(self):
    x = np.random.randn(10, 10, 10, 1)
    y = x[5, :, :]
    z = x[:, :, 2]

    t1 = data.transform.ExtractSlice()
    t2 = data.transform.ExtractSlice(axis=2)

    with self.test_session():
      x = tf.constant(x)

      self.assertAllEqual(y, t1(5, x).eval())
      self.assertAllEqual(z, t2(2, x).eval())

  def test_extract_patch(self):
    x = np.random.randn(10, 10, 10, 1)
    y = x[4:7, 4:7, 4:7]

    t = data.transform.ExtractPatch([3, 3, 3, 1])

    with self.test_session():
      self.assertAllEqual(y, t([5, 5, 5], x).eval())


if __name__ == '__main__':
  tf.test.main()
