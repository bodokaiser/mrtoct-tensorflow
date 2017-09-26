import numpy as np
import tensorflow as tf

from mrtoct import model


class SparseMovingAverageTest(tf.test.TestCase):

  def test_update_1d(self):
    indices = tf.constant([[2], [7], [8]])
    values = tf.constant([1.0, 2.0, 3.0])

    sma = model.SparseMovingAverage([10], '1d')
    op = sma.update(indices, values)
    av = sma.average()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], av.eval())
      sess.run(op)
      self.assertAllEqual([0, 0, 1, 0, 0, 0, 0, 2, 3, 0], av.eval())
      sess.run(op)
      self.assertAllEqual([0, 0, 1, 0, 0, 0, 0, 2, 3, 0], av.eval())

  def test_update_2d(self):
    indices = tf.constant([[0, 0], [1, 0], [2, 2]])
    values = tf.constant([1.0, 2.0, 3.0])

    sma = model.SparseMovingAverage([3, 3], '2d')
    op = sma.update(indices, values)
    av = sma.average()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(np.zeros([3, 3]), av.eval())
      sess.run(op)
      self.assertAllEqual([[1, 0, 0], [2, 0, 0], [0, 0, 3]], av.eval())

  def test_update_2d_patch(self):
    indices = tf.constant([[[1, 1], [1, 2]],
                           [[2, 1], [2, 2]]])
    values = tf.constant([[1.0, 2.0],
                          [3.0, 4.0]])

    sma = model.SparseMovingAverage([6, 6], '2d')
    op = sma.update(indices, values)
    av = sma.average()

    with self.test_session() as sess:
      image = np.zeros([6, 6])
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(image, av.eval())
      image[1, 1] = 1.0
      image[1, 2] = 2.0
      image[2, 1] = 3.0
      image[2, 2] = 4.0
      sess.run(op)
      self.assertAllEqual(image, av.eval())


if __name__ == '__main__':
  tf.test.main()
