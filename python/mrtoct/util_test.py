import numpy as np
import tensorflow as tf

from mrtoct import util


class UtilTest(tf.test.TestCase):

  def test_meshgrid_2d(self):
    x = util.meshgrid_2d([4, 0], [6, 2])
    y = util.meshgrid_2d([1, 10], [4, 11], 2)

    with self.assertRaises(ValueError):
      util.meshgrid_2d([0], [1, 2])
    with self.assertRaises(ValueError):
      util.meshgrid_2d([1], [1])
    with self.assertRaises(ValueError):
      util.meshgrid_2d([1, 2, 3, 4], [1, 2, 3, 4])

    with self.test_session():
      self.assertAllEqual([[[4, 0], [4, 1]], [[5, 0], [5, 1]]], x.eval())
      self.assertAllEqual([[[1, 10]], [[3, 10]]], y.eval())

  def test_meshgrid_3d(self):
    x = util.meshgrid_3d([0, 5, 0], [1, 7, 2])

    with self.test_session():
      self.assertAllEqual([[[[0, 5, 0], [0, 5, 1]],
                            [[0, 6, 0], [0, 6, 1]]]], x.eval())

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
