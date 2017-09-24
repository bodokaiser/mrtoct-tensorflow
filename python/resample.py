import argparse
import tensorflow as tf

from mrtoct import data, model, util

# volume shape
VSHAPE = [300, 340, 240]

# patch shape
PSHAPE = [32, 32, 32]

def index_to_indices(index):
    rows = tf.range(index[0]-16, index[0]+16, dtype=tf.int32)
    cols = tf.range(index[1]-16, index[1]+16, dtype=tf.int32)
    slices = tf.range(index[2]-16, index[2]+16, dtype=tf.int32)

    k, i, j = tf.meshgrid(slices, cols, rows, indexing='ij')

    return tf.stack([tf.reshape(i, [-1]), tf.reshape(j, [-1]), tf.reshape(k, [-1])])

def meshgrid():
    rows = tf.range(16, 300-16, dtype=tf.int32)
    cols = tf.range(16, 340-16, dtype=tf.int32)
    slices = tf.range(16, 240-16, dtype=tf.int32)

    k, i, j = tf.meshgrid(slices, cols, rows, indexing='ij')

    return tf.stack([tf.reshape(i, [-1]), tf.reshape(j, [-1]), tf.reshape(k, [-1])])

def resample(tfrecord, checkpoint, params):
    indices = tf.random_shuffle(tf.transpose(tf.reshape(meshgrid(), [3, -1])))

    mr_sa = model.SparseMovingAverage(VSHAPE, 'mr')
    ct_sa = model.SparseMovingAverage(VSHAPE, 'ct')

    num_indices = tf.shape(indices, out_type=tf.int64)[0]

    index_dataset = (tf.contrib.data.Dataset
                     .from_tensor_slices(indices))

    volume_dataset = (data.create_volume_dataset([tfrecord], VSHAPE)
                      .repeat(num_indices))

    patches_dataset = (tf.contrib.data.Dataset
                       .zip((volume_dataset, index_dataset))
                       .map(data.transform.extract_patch(PSHAPE))
                       .map(lambda v: tf.expand_dims(v, -1)))


    index_patch_iterator = (tf.contrib.data.Dataset
                            .zip((index_dataset.map(index_to_indices), patches_dataset)).batch(1)
                            .make_initializable_iterator())

    index, inputs = index_patch_iterator.get_next()

    outputs = model.create_generator(inputs, None, model.Mode.PREDICT, params).outputs

    aggregate = tf.group(mr_sa.apply(index, data.normalize.uncenter_zero_mean(inputs)),
                         ct_sa.apply(index, data.normalize.uncenter_zero_mean(outputs)))

    gstep = tf.assign_add(tf.train.get_or_create_global_step(), 1)

    tf.summary.image('input', tf.expand_dims(tf.expand_dims(mr_sa.average()[:, :, 100], -1), 0), 1)
    tf.summary.image('output', tf.expand_dims(tf.expand_dims(ct_sa.average()[:, :, 100], -1), 0), 1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    #saver.recover_last_checkpoints(['saved/'])


    init_op=tf.group(tf.global_variables_initializer(),
                         index_patch_iterator.initializer)

    summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter('summary')

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint('saved/.'))
        while True:
            _, s, ss = sess.run([aggregate, gstep, summary])
            writer.add_summary(ss, s)

            #print(f'step: {s}')


def main(args):
    hparams = tf.contrib.training.HParams(
        generator=model.synthgen)

    resample(args.tfrecord, args.checkpoint, hparams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord', required=True)
    parser.add_argument('--checkpoint')

    main(parser.parse_args())
