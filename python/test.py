import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct.data import transform, normalize, dataset

VSHAPE = [300, 340, 240]

def create_volume_dataset(filenames):
    return (tf.contrib.data.Dataset
                  .from_tensor_slices(filenames)
                  .flat_map(transform.filename_to_tfrecord())
                  .map(transform.tfrecord_to_tensor())
                  .map(normalize.tensor_value_range())
                  .map(normalize.tensor_shape(VSHAPE))
                  .cache())

mr_filenames = ['../data/tfrecord/re-co-mr-p005.tfrecord']
ct_filenames = ['../data/tfrecord/re-ct-p005.tfrecord']

indices = [[80, 80, 100]]

patch_dataset = (tf.contrib.data.Dataset
                 .zip((
                       dataset.create_patch_dataset(mr_filenames, indices, VSHAPE, [64, 64, 32]),
                       dataset.create_patch_dataset(ct_filenames, indices, VSHAPE, [64, 64, 16]),
                       )))

patches = patch_dataset.make_one_shot_iterator().get_next()

sess = tf.Session()

mr, ct = sess.run(patches)
