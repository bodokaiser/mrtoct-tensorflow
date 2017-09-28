import numpy as np
import tensorflow as tf

from mrtoct import data


class TransformTest(tf.test.TestCase):

  def test_compose(self):
    def fn1(i):
      return i + 1

    def fn2(i):
      return i + 2

    t = data.transform.Compose([fn1, fn2])

    self.assertEqual(3, t(0))

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

  def test_extract_patch_3d(self):
    x = np.random.randn(10, 10, 10)
    y = x[4:7, 4:7, 4:7]

    t = data.transform.ExtractPatch3D([3, 3, 3])

    with self.test_session():
      self.assertAllEqual(y, t([5, 5, 5], x).eval())


if __name__ == '__main__':
  tf.test.main()
