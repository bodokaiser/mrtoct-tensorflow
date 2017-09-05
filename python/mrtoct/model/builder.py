import enum
import collections
import tensorflow as tf

from mrtoct.model import losses
from mrtoct.model import nets

class Mode(enum.Enum):
    TRAIN = 0
    PREDICT = 2
    EVALUATE = 1

GeneratorSpec = collections.namedtuple('GeneratorSpec',
    'inputs targets outputs mode scope loss_op train_op summary_op')

DiscriminatorSpec = collections.namedtuple('DiscriminatorSpec',
    'real_score fake_score mode scope summary_op')

GenerativeAdversarialSpec = collections.namedtuple('GenerativeAdversarialSpec',
    'inputs targets outputs mode train_op summary_op')

def create_generator(inputs, targets, mode, params):
    step_op = tf.train.get_or_create_global_step()

    with tf.variable_scope('generator') as scope:
        outputs = nets.unet(inputs, params.num_filters)

        if Mode.PREDICT == mode:
            return GeneratorSpec(inputs=inputs, targets=targets,
                outputs=outputs, mode=mode, scope=scope, loss_op=None,
                train_op=None, summary_op=None)

        tf.summary.image('inputs', inputs, 1)
        tf.summary.image('targets', targets, 1)
        tf.summary.image('outputs', outputs, 1),

        with tf.name_scope('metrics'):
            mae = losses.mae(targets, outputs)
            mse = losses.mse(targets, outputs)

            tf.summary.scalar('mean_absolute_error', mae)
            tf.summary.scalar('mean_squared_error', mse)

        with tf.name_scope('loss'):
            loss_op = params.mae_weight * mae + params.mse_weight * mse

            tf.summary.scalar('total', loss_op)

        summary_op = tf.summary.merge_all()

        if Mode.EVALUATE == mode:
            return GeneratorSpec(inputs=inputs, targets=targets,
                outputs=outputs, mode=mode, scope=scope, loss_op=loss_op,
                train_op=None, summary_op=summary_op)

        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(params.learn_rate,
                params.beta1_rate).minimize(loss_op, step_op,
                scope.trainable_variables())

    return GeneratorSpec(inputs=inputs, outputs=outputs, targets=targets,
        mode=mode, scope=scope, loss_op=loss_op, train_op=train_op,
        summary_op=summary_op)

def create_discriminator(inputs, outputs, targets, mode, params):
    step = tf.train.get_or_create_global_step()

    with tf.variable_scope('discriminator') as scope:
        real_score = nets.pixel(inputs, targets, params.num_filters, False)
        fake_score = nets.pixel(inputs, outputs, params.num_filters, True)

        tf.summary.image('real_score', tf.image.convert_image_dtype(
            real_score, tf.uint8), 1)
        tf.summary.image('fake_score', tf.image.convert_image_dtype(
            fake_score, tf.uint8), 1)

    summary_op = tf.summary.merge_all()

    return DiscriminatorSpec(real_score=real_score, fake_score=fake_score,
        mode=mode, scope=scope, summary_op=summary_op)

def create_generative_adversarial_network(inputs, targets, mode, params):
    gspec = create_generator(inputs, targets, Mode.EVALUATE, params)

    if Mode.PREDICT == mode:
        return GenerativeAdversarialSpec(inputs=inputs, targets=targets,
            outputs=gspec.outputs, mode=mode, train_op=None, summary_op=None)

    dspec = create_discriminator(inputs, targets, gspec.outputs,
        Mode.EVALUATE, params)

    with tf.variable_scope('generative_adversarial'):
        gadv = losses.adv_g(dspec.fake_score)
        gloss_op = gadv + params.loss_weight * gspec.loss_op

        tf.summary.scalar('loss/gadv', gadv)
        tf.summary.scalar('loss/gloss', gloss_op)

        dloss_op = dadv = losses.adv_d(dspec.fake_score, dspec.real_score)

        tf.summary.scalar('loss/dadv', dadv)
        tf.summary.scalar('loss/dloss', dloss_op)

        step_op = tf.train.get_or_create_global_step()
        summary_op = tf.summary.merge_all()

        if Mode.EVALUATE == mode:
            return GenerativeAdversarialSpec(inputs=inputs, targets=targets,
                outputs=gspec.outputs, mode=mode, train_op=None,
                summary_op=summary_op)

        with tf.variable_scope('train'):
            gtrain_op = tf.train.AdamOptimizer(params.learn_rate,
                params.beta1_rate).minimize(gloss_op, step_op,
                gspec.scope.trainable_variables())
            dtrain_op = tf.train.AdamOptimizer(params.learn_rate,
                params.beta1_rate).minimize(dloss_op, step_op,
                dspec.scope.trainable_variables())
            train_op = tf.group(dtrain_op, gtrain_op)

    return GenerativeAdversarialSpec(inputs=inputs, targets=targets,
        outputs=gspec.outputs, mode=mode, train_op=train_op,
        summary_op=summary_op)