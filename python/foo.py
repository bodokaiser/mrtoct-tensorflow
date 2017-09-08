import tensorflow as tf

from matplotlib import pyplot as plt

from mrtoct import ioutil
from mrtoct import data

mrvol, ctvol = (data.make_zipped_dataset('../data/tfrecord')
    .make_one_shot_iterator()
    .get_next())

SIZE_MR = 32*tf.ones([3], tf.int32)
SIZE_CT = 16*tf.ones([3], tf.int32)

shape = tf.shape(mrvol)
sample = tf.stack([
    tf.random_uniform([], SIZE_MR[i]//2,shape[i]-SIZE_MR[i]//2, tf.int32)
    for i in range(shape.shape.num_elements())])

mrpatch = tf.slice(mrvol, sample-SIZE_MR//2, SIZE_MR)
ctpatch = tf.slice(ctvol, sample-SIZE_CT//2, SIZE_CT)

sess = tf.Session()

while True:
    mrv, mrp, ctv, ctp, xyz = sess.run([
        mrvol, mrpatch, ctvol, ctpatch, sample
    ])

    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(mrv[:,:,xyz[2]], vmin=-1, vmax=1)
    axes[0][1].imshow(ctv[:,:,xyz[2]], vmin=-1, vmax=1)
    axes[1][0].imshow(mrp[:,:,16], vmin=-1, vmax=1)
    axes[1][1].imshow(ctp[:,:,8], vmin=-1, vmax=1)
    plt.show()
