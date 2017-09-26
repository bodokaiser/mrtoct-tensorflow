import numpy as np
import tensorflow as tf

from mrtoct import model


class SamplerTest(tf.test.TestCase):

  def test_sample_meshgrid_3d(self):
    dataset = model.sample_meshgrid_3d([0, 0, 0], [0, 2, 2])
    index = dataset.make_one_shot_iterator().get_next()

    with self.test_session():
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.assertEqual([0, 0, 0], index.eval())
        self.assertEqual([0, 0, 1], index.eval())
        self.assertEqual([0, 1, 0], index.eval())
        self.assertEqual([0, 1, 1], index.eval())

  def test_sample_uniform_3d(self):
    dataset = model.sample_uniform_3d([0, 0, 0], [1, 2, 2], 1)
    iterator = dataset.make_initializable_iterator()

    with self.test_session() as sess:
      sess.run(iterator.initializer)

      index = iterator.get_next().eval()
      self.assertAllEqual([3], index.shape)

      for i in range(3):
        self.assertGreaterEqual(index[i], 0)
        self.assertLessEqual(index[i], 2)


if __name__ == '__main__':
  tf.test.main()
