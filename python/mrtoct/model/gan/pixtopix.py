import tensorflow as tf

from mrtoct.model import losses
from mrtoct.model.cnn import unet

xavier_init = tf.contrib.layers.xavier_initializer


def _dconv(x, num_filters):
  x = tf.layers.conv2d(x, num_filters, 4, 2, 'same',
                       kernel_initializer=xavier_init())
  return tf.nn.leaky_relu(x)


def _dfinal(x):
  x = tf.layers.conv2d(x, 1, 4, 1, 'same',
                       kernel_initializer=xavier_init())
  return tf.nn.sigmoid(x)


def discriminator_fn(y, z):
  x = tf.concat([y, z], -1)

  with tf.variable_scope('conv1'):
    x = _dconv(x, 64)
  with tf.variable_scope('conv2'):
    x = _dconv(x, 128)
  with tf.variable_scope('conv3'):
    x = _dconv(x, 256)
  with tf.variable_scope('conv4'):
    x = _dconv(x, 512)
  with tf.variable_scope('final'):
    x = _dfinal(x)

  return x


generator_fn = unet.generator_fn


def generator_loss_fn(model, **kargs):
  inputs = model.generator_inputs
  outputs = model.generated_data
  targets = model.real_data

  tf.summary.image('inputs', inputs, max_outputs=1)
  tf.summary.image('outputs', outputs, max_outputs=1)
  tf.summary.image('targets', targets, max_outputs=1)
  tf.summary.image('regress', targets - outputs, max_outputs=1)

  real_score = model.discriminator_real_outputs
  fake_score = model.discriminator_gen_outputs

  tf.summary.histogram('real_score', real_score)
  tf.summary.histogram('fake_score', fake_score)

  mae = tf.losses.absolute_difference(targets, outputs)
  mse = tf.losses.mean_squared_error(targets, outputs)
  gdl = losses.gradient_difference_loss_2d(targets, outputs)

  tf.summary.scalar('mean_absolute_error', mae)
  tf.summary.scalar('mean_squared_error', mse)
  tf.summary.scalar('gradient_difference_loss', gdl)

  adv = tf.contrib.gan.GANLoss(
      tf.contrib.gan.losses.minimax_generator_loss(model, **kargs),
      tf.contrib.gan.losses.minimax_discriminator_loss(model, **kargs))

  tf.summary.scalar('adversarial_generator_loss', adv.generator_loss)
  tf.summary.scalar('adversarial_discriminator_loss', adv.discriminator_loss)

  gan_loss = tf.contrib.gan.losses.combine_adversarial_loss(
      gan_loss=adv,
      gan_model=model,
      non_adversarial_loss=mae,
      weight_factor=0.01)

  tf.summary.scalar('generator_loss', gan_loss.generator_loss)
  tf.summary.scalar('discriminator_loss', gan_loss.discriminator_loss)

  return gan_loss.generator_loss


def discriminator_loss_fn(model, **kargs):
  return tf.contrib.gan.losses.minimax_discriminator_loss(model, **kargs)
