import tensorflow as tf

from mrtoct.patch import sample_meshgrid_3d, sample_uniform_3d


class SamplerTest(tf.test.TestCase):

  def test_sample_meshgrid_3d(self):
    indices = sample_meshgrid_3d([10, 40, 23], [11, 42, 25])

    with self.test_session():
      self.assertAllEqual([[10, 40, 23],
                           [10, 40, 24],
                           [10, 41, 23],
                           [10, 41, 24]],
                          indices.eval())

  def test_sample_uniform_3d(self):
    indices = sample_uniform_3d([0, 0, 0], [1, 2, 2], 1)

    with self.test_session():
      index = indices.eval()

      self.assertAllEqual([1, 3], index.shape)

      for i in range(3):
        self.assertGreaterEqual(index[0][i], 0)
        self.assertLessEqual(index[0][i], 2)


if __name__ == '__main__':
  tf.test.main()
