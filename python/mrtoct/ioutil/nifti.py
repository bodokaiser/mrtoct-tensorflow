import nibabel as nb
import numpy as np


def read(filename):
  """Reads a NIfTI 3D volume.

    Args:
      filename
    Returns:
      volume: numpy array in voxel space
  """
  return nb.load(filename).get_data()


def voxel_to_tensor_space(volume):
  """Rearranges volume from voxel to tensor space.

  Changes volume from Width x Height x Depth to Depth x Height x Width
  and reverses slice and row order.

  Args:
    volume: numpy array in voxel space
  Returns:
    volume: numpy array in tensor space
  """
  volume = np.transpose(volume)
  volume = np.flip(volume, 0)
  volume = np.flip(volume, 1)

  return volume


def tensor_to_voxel_space(volume):
  """Rearanges volume from tensor to voxel space.

  Args:
    volume: numpy array in tensor space
  Returns:
    volume: numpy array in voxel space
  """
  volume = np.flip(volume, 1)
  volume = np.flip(volume, 0)
  volume = np.transpose(volume)

  return volume


def orthoview(volumes):
  """Visualizes interactive orthogonal view of volumes.

  Args:
    volume: single or list of volumes in voxel space
  Returns:
    figure: `nibabel.viewers.OrthoSlicer3D` a matplotlib object
  """
  shape = np.shape(volumes)
  ndims = len(shape)

  if ndims == 3:
    return nb.viewers.OrthoSlicer3D(volumes)
  if ndims == 4:
    return nb.viewers.OrthoSlicer3D(nb.stack(volumes, -1))

  raise ValueError('volumes need to be rank 3 or 4')
