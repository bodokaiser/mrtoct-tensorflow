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
                      .map(normalize.zero_center_mean)
                      .cache())

    def extract_patches(volume):
        dataset = (tf.contrib.data.Dataset
                   .from_tensors(volume)
                   .repeat(num_indices))

        return (tf.contrib.data.Dataset
                .zip((dataset, index_dataset))
                .map(transform.extract_patch(pshape))
                .map(lambda v: tf.expand_dims(v, -1)))

    return volume_dataset.flat_map(extract_patches)
