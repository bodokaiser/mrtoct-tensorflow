import tensorflow as tf

from mrtoct import data
from mrtoct.model import losses


def cnn_model_fn(features, labels, mode, params):
  inputs = features['inputs']

  if params.data_format == 'channels_first':
    nchw_transform = data.transform.DataFormat2D('channels_first')
    nhwc_transform = data.transform.DataFormat2D('channels_last')

    outputs = params.generator_fn(nchw_transform(inputs), params.data_format)
    outputs = nhwc_transform(outputs)
  else:
    outputs = params.generator_fn(inputs, params.data_format)

  if tf.estimator.ModeKeys.PREDICT == mode:
    return tf.estimator.EstimatorSpec(mode, {
        'inputs': inputs, 'outputs': outputs})

  targets = labels['targets']

  tf.summary.image('inputs', inputs, max_outputs=1)
  tf.summary.image('outputs', outputs, max_outputs=1)
  tf.summary.image('targets', targets, max_outputs=1)
  tf.summary.image('residue', tf.abs(targets - outputs), max_outputs=1)

  mse = tf.losses.mean_squared_error(targets, outputs)
  mae = tf.losses.absolute_difference(targets, outputs)
  gdl = 1e-6 * losses.gradient_difference_loss_2d(targets, outputs)

  tf.summary.scalar('mean_squared_error', mse)
  tf.summary.scalar('mean_absolute_error', mae)
  tf.summary.scalar('gradient_difference_loss', gdl)

  total_loss = mae + gdl

  tf.summary.scalar('total_loss', total_loss)

  vars = tf.trainable_variables()

  gdl_grad = tf.global_norm(tf.gradients(gdl, vars))
  mae_grad = tf.global_norm(tf.gradients(mae, vars))
  total_grad = tf.global_norm(tf.gradients(mse, vars))

  tf.summary.scalar('gradient_difference_loss_gradient', gdl_grad)
  tf.summary.scalar('mean_absolute_error_gradient', mae_grad)
  tf.summary.scalar('total_error_gradient', total_grad)

  if tf.estimator.ModeKeys.EVAL == mode:
    return tf.estimator.EstimatorSpec(
        mode, {'outputs': outputs}, total_loss)

  optimizer = tf.train.AdamOptimizer(params.learn_rate, params.beta1_rate)

  train = optimizer.minimize(total_loss, tf.train.get_global_step())

  return tf.estimator.EstimatorSpec(
      mode, {'outputs': outputs}, total_loss, train)


def gan_model_fn(features, labels, mode, params):
  inputs = features['inputs']
  indices = features['indices']

  if params.data_format == 'channels_first':
    if inputs.shape.ndims == 4:
      in_transform = data.transform.DataFormat2D('channels_first')
      out_transform = data.transform.DataFormat2D('channels_last')
    if inputs.shape.ndims == 5:
      in_transform = data.transform.DataFormat3D('channels_first')
      out_transform = data.transform.DataFormat3D('channels_last')
  else:
    in_transform = out_transform = lambda x: x

  def generator_fn(x):
    return params.generator_fn(x, params.data_format)

  if tf.estimator.ModeKeys.PREDICT == mode:
    with tf.variable_scope('Generator'):
      outputs = out_transform(generator_fn(in_transform(inputs)))

    return tf.estimator.EstimatorSpec(
        mode, {'inputs': inputs, 'outputs': outputs, 'indices': indices})

  targets = labels['targets']

  def discriminator_fn(x, y):
    return params.discriminator_fn(x, y, params.data_format)

  gan_model = tf.contrib.gan.gan_model(
      generator_fn=generator_fn,
      discriminator_fn=discriminator_fn,
      real_data=in_transform(targets),
      generator_inputs=in_transform(inputs))

  outputs = out_transform(gan_model.generated_data)

  gan_loss = tf.contrib.gan.gan_loss(
      model=gan_model,
      generator_loss_fn=params.generator_loss_fn,
      discriminator_loss_fn=params.discriminator_loss_fn,
  )

  if inputs.shape.ndims == 4:
    tf.summary.image('inputs', inputs, max_outputs=1)
    tf.summary.image('outputs', outputs, max_outputs=1)
    tf.summary.image('targets', targets, max_outputs=1)
    tf.summary.image('residue', targets - outputs, max_outputs=1)

  if inputs.shape.ndims == 5:
    tf.summary.image('inputs', inputs[:, 16], max_outputs=1)
    tf.summary.image('outputs', outputs[:, 8], max_outputs=1)
    tf.summary.image('targets', targets[:, 8], max_outputs=1)
    tf.summary.image('residue', targets[:, 8] - outputs[:, 8], max_outputs=1)

  with tf.name_scope('loss'):
    mae = tf.norm(targets - outputs, ord=1)
    mse = tf.norm(targets - outputs, ord=2)

    if inputs.shape.ndims == 4:
      gdl = losses.gradient_difference_loss_2d(targets, outputs)
    if inputs.shape.ndims == 5:
      gdl = losses.gradient_difference_loss_3d(targets, outputs)

    tf.summary.scalar('mean_squared_error', mse)
    tf.summary.scalar('mean_absolute_error', mae)
    tf.summary.scalar('gradient_difference_loss', gdl)

    loss = 3 * mae + gdl

    tf.summary.scalar('total_loss', loss)

    vars = tf.trainable_variables()

    gdl_grad = tf.global_norm(tf.gradients(gdl, vars))
    mae_grad = tf.global_norm(tf.gradients(mae, vars))
    mse_grad = tf.global_norm(tf.gradients(mse, vars))

    tf.summary.scalar('gradient_difference_loss_gradient', gdl_grad)
    tf.summary.scalar('mean_absolute_error_gradient', mae_grad)
    tf.summary.scalar('mean_squared_error_gradient', mse_grad)

    real_score = gan_model.discriminator_real_outputs
    fake_score = gan_model.discriminator_gen_outputs

    tf.summary.histogram('real_score', real_score)
    tf.summary.histogram('fake_score', fake_score)

    gan_loss = tf.contrib.gan.losses.combine_adversarial_loss(
        gan_loss=gan_loss,
        gan_model=gan_model,
        non_adversarial_loss=loss,
        weight_factor=params.weight_factor,
    )

  with tf.name_scope('train'):
    generator_optimizer = tf.train.AdamOptimizer(
        params.learn_rate, params.beta1_rate)
    discriminator_optimizer = tf.train.AdamOptimizer(
        params.learn_rate, params.beta1_rate)

    train = tf.contrib.gan.gan_train_ops(
        model=gan_model,
        loss=gan_loss,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer)
    train_op = tf.group(*list(train))

  return tf.estimator.EstimatorSpec(
      mode, {'outputs': outputs}, loss, train_op)
