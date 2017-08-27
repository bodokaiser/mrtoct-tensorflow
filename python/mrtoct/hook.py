import tensorflow as tf


class TrainValidFeedHook(tf.train.SessionRunHook):

    def __init__(self, iterator, n_steps):
        self.handle = iterator.get_handle()
        self.train_handle = iterator.get_train_handle()
        self.valid_handle = iterator.get_valid_handle()
        self.n_steps = n_steps

    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()

    def after_create_session(self, sess, coord):
        self.last_step, self.train_handle, self.valid_handle = sess.run([
            self.global_step, self.train_handle, self.valid_handle])

    def before_run(self, run_context):
        train = self.last_step // self.n_steps % 2 == 0

        return tf.train.SessionRunArgs(self.global_step, {
            self.handle: self.train_handle if train else self.valid_handle})

    def after_run(self, run_context, run_values):
        self.last_step = run_values.results