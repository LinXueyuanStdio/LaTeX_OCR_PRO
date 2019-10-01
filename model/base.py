import os
import time

import tensorflow as tf

from .utils.general import init_dir, get_logger


class BaseModel(object):
    """Generic class for tf models"""

    def __init__(self, config, dir_output):
        """Defines self._config

        Args:
            config: (Config instance) class with hyper parameters, vocab and embeddings

        """
        self._config = config
        self._dir_output = dir_output
        init_dir(self._dir_output)
        self.logger = get_logger(self._dir_output + "model.log")
        config.show(fun = self.logger.info)
        tf.reset_default_graph()  # save guard if previous model was defined

    def build_train(self, config=None):
        """To overwrite with model-specific logic"""
        raise NotImplementedError

    def build_pred(self, config=None):
        """Similar to build_train but no need to define train_op"""
        raise NotImplementedError

    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        self.sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)
        dir_model = self._dir_output + "model_weights/"
        init_dir(dir_model)
        self.ckeck_point = tf.train.latest_checkpoint(dir_model)
        print("checkpoint", self.ckeck_point)
        self.startepoch = 0
        if self.ckeck_point != None:
            self.saver.restore(self.sess, self.ckeck_point)
            idx = self.ckeck_point.find("-")
            self.startepoch = int(self.ckeck_point[idx+1:])
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! find a checkpoint, load epoch ", self.startepoch)
        self._add_summary()  # tensorboard 可视化

    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self, epoch):
        """Saves session"""
        # check dir one last time
        dir_model = self._dir_output + "model_weights/"
        init_dir(dir_model)

        self.logger.info("- Saving model...")
        self.saver.save(self.sess, dir_model+"model.cpkt", global_step=epoch)
        self.logger.info("- Saved model in {}".format(dir_model))

    def save_debug_session(self, epoch, i):
        """Saves session"""
        # check dir one last time
        dir_model = self._dir_output + "debug_model_weights/"
        init_dir(dir_model)

        self.logger.info("- Saving model...")
        self.saver.save(self.sess, dir_model+"model_"+str(i)+".cpkt", global_step=epoch)
        self.logger.info("- Saved model in {}".format(dir_model))

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def _add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self._dir_output, self.sess.graph)

    def train(self, config, train_set, val_set, lr_schedule):
        """Global training procedure

        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic including the lr_schedule update must be done in
        self.run_epoch

        Args:
            config: Config instance contains params as attributes
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            best_score: (float)

        """
        best_score = None

        for epoch in range(config.n_epochs):
            if epoch < self.startepoch:
                continue

            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch+1, config.n_epochs))

            # epoch
            score = self._run_train(config, train_set, val_set, epoch, lr_schedule)

            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(best_score))
                self.save_session(epoch)
            if lr_schedule.stop_training:
                self.logger.info("- Early Stopping.")
                break

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}, learning rate: {:04.5f}".format(toc-tic, lr_schedule.lr))

        return best_score

    def _run_train(self, config, train_set, val_set, epoch, lr_schedule):
        """Model_specific method to overwrite

        Performs an epoch of training

        Args:
            config: Config
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest score

        """
        raise NotImplementedError

    def evaluate(self, config, test_set):
        """Evaluates model on test set

        Calls method run_evaluate on test_set and takes care of logging

        Args:
            config: Config
            test_set: instance of class Dataset

        Return:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        self.logger.info("- Evaluating...")
        scores = self._run_evaluate(config, test_set)
        msg = " || ".join([" {} is {:04.2f} ".format(k, v) for k, v in scores.items()])
        self.logger.info("- Eval: {}".format(msg))

        return scores

    def _run_evaluate(self, config, test_set):
        """Model-specific method to overwrite

        Performs an epoch of evaluation

        Args:
            config: Config
            test_set: Dataset instance

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        raise NotImplementedError
