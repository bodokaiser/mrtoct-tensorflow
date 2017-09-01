import tensorflow as tf

def model_fn(features, labels, mode, params):
    inputs, targets = features['inputs'], labels['targets']

    with tf.name_scope('model'):
        outputs = params['generator'](inputs)

        tf.summary.image('inputs', inputs)
        tf.summary.image('outputs', outputs)
        tf.summary.image('targets', targets)

    if tf.estimator.ModeKeys.PREDICT == mode:
        return tf.estimator.EstimatorSpec(mode, {'outputs': outputs})

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(targets, outputs)

        tf.summary.scalar('mse', loss)

    with tf.name_scope('train'):
        step = tf.train.get_or_create_global_step()
        train = tf.train.AdamOptimizer(params['lr']).minimize(loss, step)

    return tf.estimator.EstimatorSpec(mode, {'outputs': outputs}, loss, train,
        {'mse': tf.metrics.mean_squared_error(targets, outputs)})

def make_input_fn(dataset):
    def input_fn():
        inputs, targets = dataset.make_one_shot_iterator().get_next()

        return {'inputs': inputs}, {'targets': targets}

    return input_fn