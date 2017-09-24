import tensorflow as tf

from mrtoct import model, data

# number of indices to sample
NUM_SAMPLES = 10000

# volume shape
VSHAPE = [300, 340, 240]

# input patch shape
PSHAPE_INPUT = [32, 32, 32]

# output patch shape
PSHAPE_OUTPUT = [32, 32, 32]


params = tf.contrib.training.HParams(
    learn_rate=1e-6,
    beta1_rate=5e-1,
    mse_weight=1.00,
    mae_weight=0.00,
    gdl_weight=1.00,
    adv_weight=0.50,
    generator=model.synthgen,
    discriminator=model.synthdisc)

mr_filenames = tf.gfile.Glob('../data/tfrecord/re-co-mr-p005.tfrecord')
ct_filenames = tf.gfile.Glob('../data/tfrecord/re-ct-p005.tfrecord')

assert len(mr_filenames) == len(ct_filenames), 'invalid volume pair'
assert len(mr_filenames) > 0, 'no volumes found'

#indices = tf.random_shuffle(util.meshgrid(
#    PSHAPE_INPUT, tf.subtract(VSHAPE, PSHAPE_INPUT), 2, 3))

indices = tf.stack([
        tf.random_uniform([NUM_SAMPLES], minval=PSHAPE_OUTPUT[0],
                          maxval=VSHAPE[0]-PSHAPE_OUTPUT[0], dtype=tf.int32),
        tf.random_uniform([NUM_SAMPLES], minval=PSHAPE_OUTPUT[1],
                          maxval=VHSAPE[1]-PSHAPE_OUTPUT[1], dtype=tf.int32),
        tf.random_uniform([NUM_SAMPLES], minval=PSHAPE_OUTPUT[2],
                          maxval=VSHAPE[2]-PSHAPE_OUTPUT[2], dtype=tf.int32)], 1)

input_patch_dataset = data.create_patch_dataset(
    mr_filenames, indices, VSHAPE, PSHAPE_INPUT)
target_patch_dataset = data.create_patch_dataset(
    ct_filenames, indices, VSHAPE, PSHAPE_OUTPUT)

patch_dataset = (tf.contrib.data.Dataset
                 .zip((input_patch_dataset, target_patch_dataset))
                 .batch(10).repeat())
patch_iterator = patch_dataset.make_initializable_iterator()

input_patches, target_patches = patch_iterator.get_next()

spec = model.create_generative_adversarial_network(input_patches,
    target_patches, model.Mode.TRAIN, params)

step = tf.train.get_global_step()

output_patches = spec.outputs

uncenter = data.normalize.uncenter_zero_mean

tf.summary.image('input', uncenter(input_patches[:, :, :, 0]), 1)
tf.summary.image('output', uncenter(output_patches[:, :, :, 0]), 1)
tf.summary.image('target', uncenter(target_patches[:, :, :, 0]), 1)

initialize = tf.group(tf.global_variables_initializer(),
                      patch_iterator.initializer)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

scaffold = tf.train.Scaffold(init_op=initialize)

with tf.train.MonitoredTrainingSession(config=config,
                                       scaffold=scaffold,
                                       checkpoint_dir='results',
                                       save_checkpoint_secs=None,
                                       save_summaries_steps=10) as sess:
    while not sess.should_stop():
        s, _ = sess.run([step, spec.train_op])

        if s % (NUM_SAMPLES // 10 - 1) == 0:
            sess.run(patch_iterator.initializer)

        print(f'step: {s}')
