import numpy as np
import tensorflow as tf

from mrtoct import model, util


class PatchAggregatorTest(tf.test.TestCase):

  def setUp(self):
    self.index_placeholder = tf.placeholder(tf.int32, [2, 3, 3, 3])
    self.patch_placeholder = tf.placeholder(tf.float32, [2, 3, 3, 3])
    self.patch_aggregator = model.PatchAggregator3D([10, 10, 10], 'foo')

  def test_update(self):
    p1 = np.random.randint(0, 256, [3, 3, 3])
    p2 = np.random.randint(0, 256, [3, 3, 3])

    av = np.zeros([10, 10, 10])
    av[0:3, 0:3, 0:3] += p1
    av[2:5, 0:3, 0:3] += p2
    av[2:3, 0:3, 0:3] /= 2

    op = self.patch_aggregator.update(self.index_placeholder,
                                      self.patch_placeholder)
    with self.test_session() as sess:
      i1 = util.meshgrid_3d([0, 0, 0], [3, 3, 3]).eval()
      i2 = util.meshgrid_3d([2, 0, 0], [5, 3, 3]).eval()

      sess.run(op, feed_dict={
          self.index_placeholder: i1,
          self.patch_placeholder: p1})
      sess.run(op, feed_dict={
          self.index_placeholder: i2,
          self.patch_placeholder: p2})

      self.assertAllEqual(av, self.patch_aggregator.average().eval())


if __name__ == '__main__':
  tf.test.main()
