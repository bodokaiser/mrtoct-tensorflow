import numpy as np
import tensorflow as tf

from mrtoct import util


class UtilTest(tf.test.TestCase):

  def test_meshgrid_2d(self):
    x = util.meshgrid_2d([0, 0], [2, 2])
    y = util.meshgrid_2d([10, 10], [14, 14], 2)

    with self.assertRaises(ValueError):
      util.meshgrid_2d([0], [1, 2])
    with self.assertRaises(ValueError):
      util.meshgrid_2d([1], [1])
    with self.assertRaises(ValueError):
      util.meshgrid_2d([1, 2, 3, 4], [1, 2, 3, 4])

    with self.test_session():
      self.assertAllEqual([[[0, 0], [0, 1]],
                           [[1, 0], [1, 1]]], x.eval())
      self.assertAllEqual([[[10, 10], [10, 12]],
                           [[12, 10], [12, 12]]], y.eval())

  def test_meshgrid_3d(self):
    x = util.meshgrid_3d([0, 0, 0], [1, 2, 2])

    with self.test_session():
      self.assertAllEqual([[[[0, 0, 0], [0, 0, 1]],
                            [[0, 1, 0], [0, 1, 1]]]], x.eval())

  def test_spatial_gradient_3d(self):
    volume = np.random.randint(0, 10, [5, 10, 10, 10, 3])
    gradx = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
    grady = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
    gradz = volume[:, 1:, :, :, :] - volume[:, :-1, :, :, :]

    with self.test_session():
      dz, dy, dx = util.spatial_gradient_3d(volume)

      self.assertAllEqual(gradz, dz.eval())
      self.assertAllEqual(grady, dy.eval())
      self.assertAllEqual(gradx, dx.eval())


if __name__ == '__main__':
  tf.test.main()
