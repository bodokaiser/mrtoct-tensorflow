import tensorflow as tf

from mrtoct import util


class UtilTest(tf.test.TestCase):

  def testMeshgrid(self):
    with self.test_session():
      x = util.meshgrid([0, 0], [2, 2])
      y = util.meshgrid([10, 10], [14, 14], 2)
      z = util.meshgrid([0, 0, 0], [1, 2, 2])

      with self.assertRaises(ValueError):
        util.meshgrid([0], [1, 2])
      with self.assertRaises(ValueError):
        util.meshgrid([1], [1])
      with self.assertRaises(ValueError):
        util.meshgrid([1, 2, 3, 4], [1, 2, 3, 4])

      self.assertAllEqual([[[0, 0], [0, 1]],
                           [[1, 0], [1, 1]]], x.eval())
      self.assertAllEqual([[[10, 10], [10, 12]],
                           [[12, 10], [12, 12]]], y.eval())
      self.assertAllEqual(
          [[
              [[0, 0, 0], [0, 0, 1]],
              [[0, 1, 0], [0, 1, 1]],
          ]], z.eval())


if __name__ == '__main__':
  tf.test.main()
