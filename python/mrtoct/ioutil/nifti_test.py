import tensorflow as tf

from mrtoct import ioutil


class NIfTITest(tf.test.TestCase):

  def test_read_nifti(self):
    volume = ioutil.read_nifti('../data/nifti/re-ct-p001.nii')

    self.assertAllEqual((250, 320, 161), volume.shape)


if __name__ == '__main__':
  tf.test.main()
