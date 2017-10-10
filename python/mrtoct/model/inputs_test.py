import tensorflow as tf

from mrtoct import model


class InputsTest(tf.test.TestCase):

  def test_train_patch_input_fn(self):
    inputs, targets = model.train_patch_input_fn(
        inputs_path='../data/tfrecord/training/mr.tfrecord',
        targets_path='../data/tfrecord/training/mr.tfrecord',
        volume_shape=[260, 340, 360, 1],
        inputs_shape=[32, 32, 32, 1],
        targets_shape=[16, 16, 16, 1],
        batch_size=10)

    with self.test_session() as session:
      for i in range(5):
        x, y = session.run([inputs, targets])

        self.assertAllEqual(x[:, 8:24, 8:24, 8:24], y)


if __name__ == '__main__':
  tf.test.main()
