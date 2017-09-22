import tensorflow as tf

from mrtoct import util, model, data


# volume shape
VSHAPE = [300, 340, 240, 1]

# input patch shape
PSHAPE_INPUT = [32, 32, 32, 1]

# output patch shape
PSHAPE_OUTPUT = [16, 16, 16, 1]

params = tf.contrib.training.HParams(
    learn_rate=1e-6,
    beta1_rate=5e-1,
    mse_weight=1.00,
    mae_weight=0.00,
    gdl_weight=1.00,
    adv_weight=0.50,
    generator=model.synthgen,
    discriminator=model.synthdisc)

mr_filenames = tf.gfile.Glob('../data/tfrecord/re-co-mr-p001.tfrecord')
ct_filenames = tf.gfile.Glob('../data/tfrecord/re-ct-p001.tfrecord')

indices = tf.random_shuffle(util.meshgrid(
    PSHAPE_INPUT, tf.subtract(VSHAPE, PSHAPE_INPUT), 2, 3))

input_patch_dataset = data.create_patch_dataset(
    mr_filenames, indices, VSHAPE, PSHAPE_INPUT)
target_patch_dataset = data.create_patch_dataset(
    ct_filenames, indices, VSHAPE, PSHAPE_OUTPUT)

patch_dataset = (tf.contrib.data.Dataset
                 .zip((input_patch_dataset, target_patch_dataset))
                 .batch(10).repeat())
                 #.shuffle(1000).batch(10).repeat())
patch_iterator = patch_dataset.make_initializable_iterator()

input_indices_patches, target_indices_patches = patch_iterator.get_next()

input_indices, input_patches = input_indices_patches
target_indices, target_patches = target_indices_patches

spec = model.create_generator(input_patches, target_patches, model.Mode.TRAIN,
                              params)

step = tf.train.get_global_step()

output_patches = spec.outputs

tf.summary.image('patches/input', input_patches[:, :, :, 0], 3)
tf.summary.image('patches/output', output_patches[:, :, :, 0], 3)
tf.summary.image('patches/target', target_patches[:, :, :, 0], 3)

initialize = tf.group(tf.global_variables_initializer(),
                      patch_iterator.initializer)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

scaffold = tf.train.Scaffold(init_op=initialize)

with tf.train.MonitoredTrainingSession(config=config,
                                       scaffold=scaffold,
                                       checkpoint_dir='/tmp/mrtoct',
                                       save_checkpoint_secs=None,
                                       save_summaries_steps=10) as sess:
    while not sess.should_stop():
        s, _ = sess.run([step, spec.train_op])

        print(f'step: {s}')
