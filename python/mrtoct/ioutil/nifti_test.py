import numpy as np
import tensorflow as tf

from mrtoct import ioutil


class NIfTITest(tf.test.TestCase):

  def test_read_nifti(self):
    volume = ioutil.read_nifti('../data/nifti/re-ct-p001.nii')

    self.assertAllEqual((250, 320, 161), volume.shape)

  def test_voxel_to_tensor_space(self):
    vspace = np.random.randn(250, 320, 160)
    tspace = ioutil.voxel_to_tensor_space(vspace)

    self.assertAllEqual((160, 320, 250, 1), tspace.shape)

  def test_tensor_to_voxel_space(self):
    tspace = np.random.randn(160, 320, 250, 1)
    vspace = ioutil.tensor_to_voxel_space(tspace)

    self.assertAllEqual((250, 320, 160), vspace.shape)
    self.assertAllEqual(tspace, ioutil.voxel_to_tensor_space(vspace))


if __name__ == '__main__':
  tf.test.main()
