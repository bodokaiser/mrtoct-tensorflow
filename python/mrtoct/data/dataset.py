import tensorflow as tf

from mrtoct.data import normalize, transform


def create_patch_dataset(filenames, indices, vshape, pshape):
    num_indices = tf.shape(indices, out_type=tf.int64)[0]

    index_dataset = (tf.contrib.data.Dataset
                     .from_tensor_slices(indices))

    volume_dataset = (tf.contrib.data.Dataset
                      .from_tensor_slices(filenames)
                      .flat_map(transform.filename_to_tfrecord())
                      .map(transform.tfrecord_to_tensor())
                      .map(normalize.tensor_value_range())
                      .map(normalize.tensor_shape(vshape))
                      .enumerate()
                      .cache())

    def extract_patches(id, volume):
        dataset = (tf.contrib.data.Dataset
                   .from_tensors((id, volume))
                   .repeat(num_indices))

        return (tf.contrib.data.Dataset
                .zip((dataset, index_dataset))
                .map(transform.extract_patch(pshape)))

    return volume_dataset.flat_map(extract_patches)
