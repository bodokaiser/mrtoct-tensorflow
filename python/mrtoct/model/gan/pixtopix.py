import tensorflow as tf

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


def _gfinal(x):
  x = tf.layers.conv2d_transpose(x, 1, 3, 1, 'same',
                                 kernel_initializer=xavier_init())
  return tf.nn.tanh(x)


def generator_fn(x):
  with tf.variable_scope('encode1'):
    enc1 = unet.encode(x, 64, batch_norm=False)
  with tf.variable_scope('encode2'):
    enc2 = unet.encode(enc1, 128)
  with tf.variable_scope('encode3'):
    enc3 = unet.encode(enc2, 256)
  with tf.variable_scope('encode4'):
    enc4 = unet.encode(enc3, 512)
  with tf.variable_scope('encode5'):
    enc5 = unet.encode(enc4, 512)

  with tf.variable_scope('decode5'):
    dec5 = unet.decode(enc5, 512, dropout=True)
  with tf.variable_scope('decode4'):
    dec4 = unet.decode(tf.concat([dec5, enc4], -1), 512, dropout=True)
  with tf.variable_scope('decode3'):
    dec3 = unet.decode(tf.concat([dec4, enc3], -1), 256)
  with tf.variable_scope('decode2'):
    dec2 = unet.decode(tf.concat([dec3, enc2], -1), 128)
  with tf.variable_scope('decode1'):
    dec1 = unet.decode(tf.concat([dec2, enc1], -1), 64)

  with tf.variable_scope('final'):
    return _gfinal(dec1)


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


def optimizer_fn():
  return tf.train.AdamOptimizer(2e-4, 5e-1)


def generator_loss_fn(model, **kargs):
  mae = tf.losses.mean_absolute_error(model.real_data,
                                      model.generated_data)

  adv = tf.contrib.gan.GANLoss(
      tf.contrib.gan.losses.minimax_generator_loss(model, **kargs),
      tf.contrib.gan.losses.minimax_discriminator_loss(model, **kargs))

  gan_loss = tf.contrib.gan.losses.combine_adversarial_loss(
      gan_loss=adv,
      gan_model=model,
      non_adversarial_loss=mae,
      weight_factor=0.01)

  return gan_loss.generator_loss


def discriminator_loss_fn(model, **kargs):
  return tf.contrib.gan.losses.minimax_discriminator_loss(model, **kargs)
