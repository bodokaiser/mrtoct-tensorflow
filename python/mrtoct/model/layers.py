import tensorflow as tf


class Conv2D(tf.layers.Conv2D):
  """Same as tf.layers.Conv2D but with better defaults."""

  def __init__(self, *args, **kwargs):
    init = tf.contrib.layers.xavier_initializer()

    super().__init__(*args, kernel_initializer=init, padding='same', **kwargs)


class Conv3D(tf.layers.Conv3D):
  """Same as tf.layers.Conv3D but with better defaults."""

  def __init__(self, num_filters, kernel_size, padding='same'):
    init = tf.contrib.layers.xavier_initializer()

    super().__init__(num_filters, kernel_size,
                     padding=padding, kernel_initializer=init)


class Conv2DTranspose(tf.layers.Conv2DTranspose):
  """Same as tf.layers.Conv2DTranspose but with better defaults."""

  def __init__(self, *args, **kwargs):
    init = tf.contrib.layers.xavier_initializer()

    super().__init__(*args, kernel_initializer=init, padding='same', **kwargs)


Input = tf.layers.Input
Dense = tf.layers.Dense
Dropout = tf.layers.Dropout
BatchNorm = tf.layers.BatchNormalization
MaxPool3D = tf.layers.MaxPooling3D
Network = tf.keras.models.Model

Activation = tf.keras.layers.Activation
LeakyReLU = tf.keras.layers.LeakyReLU
Concatenate = tf.keras.layers.Concatenate
