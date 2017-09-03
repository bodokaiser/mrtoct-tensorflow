import tensorflow as tf

# stabilizes logarithmic loss
EPSILON = 1e-12

def model_fn(features, labels, mode, params):
    inputs, targets = features['inputs'], labels['targets']

    with tf.variable_scope('generator'):
        outputs = params['generator'](inputs)

        with tf.name_scope('loss'):
            l1loss = tf.reduce_mean(tf.abs(targets - outputs))
            l2loss = tf.reduce_mean(tf.square(targets - outputs))
            gloss = l1loss

            tf.summary.scalar('l1', l1loss)
            tf.summary.scalar('l2', l2loss)
            tf.summary.scalar('loss', gloss)

    labels = {'outputs': outputs}

    if tf.estimator.ModeKeys.PREDICT == mode:
        return tf.estimator.EstimatorSpec(mode, labels)

    tf.summary.image('inputs', inputs, 1)
    tf.summary.image('outputs', outputs, 1)
    tf.summary.image('targets', targets, 1)

    if params['discriminator'] is not None:
        with tf.name_scope('discriminator'):
            with tf.name_scope('real'):
                with tf.variable_scope('discriminator'):
                    rscore = params['discriminator'](inputs, targets)

                    tf.summary.histogram('score', rscore)

            with tf.name_scope('fake'):
                with tf.variable_scope('discriminator', reuse=True):
                    fscore = params['discriminator'](inputs, outputs)

                    tf.summary.histogram('score', fscore)

            with tf.name_scope('loss'):
                dgloss = tf.reduce_mean(-tf.log(rscore + EPSILON)
                    - tf.log(1 - fscore + EPSILON))
                dloss = dgloss

                tf.summary.scalar('gan', dloss)
                tf.summary.scalar('loss', dloss)

            with tf.name_scope('train'):
                dvars = [v for v in tf.trainable_variables()
                    if v.name.startswith('discriminator')]
                doptim = tf.train.AdamOptimizer(params['lr'])
                dgrads = doptim.compute_gradients(dloss, dvars)
                dtrain = doptim.apply_gradients(dgrads,
                    tf.train.get_or_create_global_step())

        with tf.variable_scope('generator/loss', reuse=True):
            ggloss = tf.reduce_mean(-tf.log(fscore + EPSILON))
            gloss += ggloss

            tf.summary.scalar('gan', ggloss)
    else:
        dtrain = tf.no_op()

    with tf.name_scope('generator/train'):
        gvars = [v for v in tf.trainable_variables()
            if v.name.startswith("generator")]
        goptim = tf.train.AdamOptimizer(params['lr'])
        ggrads = goptim.compute_gradients(gloss, gvars)
        gtrain = goptim.apply_gradients(ggrads,
            tf.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(mode, labels,
        tf.add(dloss, gloss), tf.group(dtrain, gtrain), {
            'mse': tf.metrics.mean_squared_error(targets, outputs),
            'mae': tf.metrics.mean_absolute_error(targets, outputs),
        })
