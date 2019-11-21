import sys
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from PIL import Image
import random

import model.components.attention_mechanism
from model.base import BaseModel
from model.decoder import Decoder
from model.encoder2 import Encoder
from model.evaluation.text import score_files, truncate_end, write_answers
from model.utils.general import Config, Progbar, minibatches
from model.utils.image import pad_batch_images
from model.utils.text import pad_batch_formulas


class Img2SeqModel(BaseModel):
    """Specialized class for Img2Seq Model"""

    def __init__(self, config, dir_output, vocab):
        """
        Args:
            config: Config instance defining hyperparams
            vocab: Vocab instance defining useful vocab objects like tok_to_id

        """
        super(Img2SeqModel, self).__init__(config, dir_output)
        model.components.attention_mechanism.ctx_vector = []
        self._vocab = vocab

    def build_train(self, config):
        """Builds model"""
        self.logger.info("Building model...")
        self.build_base_component()
        self.add_optimizer(config.lr_method, self.lr, self.loss, config.clip)
        self.init_session()
        self.logger.info("- done.")

    def build_pred(self):
        self.logger.info("Building model...")
        self.build_base_component()
        self.init_session()
        self.logger.info("- done.")

    def build_base_component(self):
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(), name='lr')  # learning rate
        self.dropout = tf.placeholder(tf.float32, shape=(),   name='dropout')

        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, None, None, 1),  name='img')  # (N, H, W, C)，这里C=1，因为是灰度图
        self.formula = tf.placeholder(tf.int32, shape=(None, None),  name='formula')  # (N, formula_tokens)
        self.formula_length = tf.placeholder(tf.int32, shape=(None, ),   name='formula_length')  # (N, 1)

        # self.pred_train, self.pred_test
        # tensorflow 只有静态计算图，只好同时把 train 和 test 部分的计算图都建了
        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok, self._vocab.id_end)
        encoded_img = self.encoder(self.img, self.dropout)
        train, test = self.decoder(encoded_img, self.formula, self.dropout)
        self.pred_train = train
        self.pred_test = test

        # self.loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_train, labels=self.formula)
        mask = tf.sequence_mask(self.formula_length)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # to compute perplexity for test
        self.ce_words = tf.reduce_sum(losses)  # sum of CE for each word
        self.n_words = tf.reduce_sum(self.formula_length)  # number of words

        # tensorboard
        tf.summary.image("img", self.img)
        tf.summary.scalar("learning_rate", self.lr)
        tf.summary.scalar("dropout", self.dropout)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("sum_of_CE_for_each_word", self.ce_words)
        tf.summary.scalar("number_of_words", self.n_words)

    def add_optimizer(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping


        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("optimize"):
            # sgd method 优化器
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            # for batch norm beta gamma
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if clip > 0:
                    # gradient clipping if clip is positive, defaults -1
                    # 梯度裁剪 (gradient clipping) 用于解决梯度爆炸 (gradient explosion) 问题
                    # 另外，与 grdient explosion 相反的问题 gradient vanishing 的做法跟 grdient explosion 不同
                    # gradient vanishing 一般是采用 LSTM 或 GRU 这类有记忆的 RNN 单元
                    grads, vs = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, vs))
                else:
                    self.train_op = optimizer.minimize(loss)

    def _get_feed_dict(self, img, formula=None, lr=None, dropout=1):
        """Returns a dict 网络的输入"""
        img = pad_batch_images(img)

        fd = {
            self.img: img,
            self.dropout: dropout,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula, self._vocab.id_pad, self._vocab.id_end)
            # print img.shape, formula.shape
            fd[self.formula] = formula
            fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr

        return fd

    def _run_train(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training

        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest score

        """
        # logging
        batch_size = config.batch_size
        train_set.shuffle()
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        prog = Progbar(nbatches)

        # iterate over dataset
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            # get feed dict
            fd = self._get_feed_dict(img, formula=formula, lr=lr_schedule.lr, dropout=config.dropout)
            # 来试试随机的 dropout
            # random_dropout = 0.5 + random.random() * 0.5
            # fd = self._get_feed_dict(img, formula=formula, lr=lr_schedule.lr, dropout=random_dropout)

            # update step
            _, loss_eval = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            prog.update(i + 1, [("loss", loss_eval), ("perplexity", np.exp(loss_eval)), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

            # 生成summary
            if (i+1) % 10 == 0:
                summary_str = self.sess.run(self.merged, feed_dict=fd)
                self.file_writer.add_summary(summary_str, epoch)  # 将summary 写入文件

            # if (i+1) % 100 == 0:
            #     # 太慢了，读了 100 批次后就保存先，保存的权重要用于调试 attention
            #     self.save_debug_session(epoch, i)

        # logging
        self.logger.info("- Training: {}".format(prog.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun = self.logger.info)

        # evaluation
        config_eval = Config({
            "dir_answers": self._dir_output + "formulas_val/",
            "batch_size": config.batch_size + 20
        })
        scores = self.evaluate(config_eval, val_set)
        score = scores["perplexity"] + (scores["ExactMatchScore"] + scores["BLEU-4"] + scores["EditDistance"]) / 10
        lr_schedule.update(score=score)

        return score

    def _run_evaluate(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            config: (Config) with batch_size and dir_answers

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        files, perp = self.write_prediction(config, test_set)
        scores = score_files(files[0], files[1])
        scores["perplexity"] = perp

        return scores

    def write_prediction(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            config: (Config) with batch_size and dir_answers
            test_set:(Dataset) instance

        Returns:
            files: (list) of path to files
            perp: (float) perplexity on test set

        """
        # initialize containers of references and predictions
        if self._config.decoding == "greedy":
            refs, hyps = [], [[]]
        elif self._config.decoding == "beam_search":
            refs, hyps = [], [[] for i in range(self._config.beam_size)]

        nbatches = (len(test_set) + config.batch_size - 1) // config.batch_size
        prog = Progbar(nbatches)
        n_words, ce_words = 0, 0  # sum of ce for all words + nb of words
        for i, (img, formula) in enumerate(minibatches(test_set, config.batch_size)):
            fd = self._get_feed_dict(img, formula=formula, dropout=1)
            ce_words_eval, n_words_eval, ids_eval = self.sess.run([self.ce_words, self.n_words, self.pred_test.ids], feed_dict=fd)

            if self._config.decoding == "greedy":
                ids_eval = np.expand_dims(ids_eval, axis=1)
            elif self._config.decoding == "beam_search":
                ids_eval = np.transpose(ids_eval, [0, 2, 1])
            n_words += n_words_eval
            ce_words += ce_words_eval

            for form, preds in zip(formula, ids_eval):
                refs.append(form)
                for j, pred in enumerate(preds):
                    hyps[j].append(pred)

            prog.update(i + 1, [("perplexity", - np.exp(ce_words / float(n_words)))])

        files = write_answers(refs, hyps, self._vocab.id_to_tok, config.dir_answers, self._vocab.id_end)

        perp = - np.exp(ce_words / float(n_words))

        return files, perp

    def predict_batch(self, images):
        if self._config.decoding == "greedy":
            hyps = [[]]
        elif self._config.decoding == "beam_search":
            hyps = [[] for i in range(self._config.beam_size)]

        fd = self._get_feed_dict(images, dropout=1)
        ids_eval, = self.sess.run([self.pred_test.ids], feed_dict=fd)

        if self._config.decoding == "greedy":
            ids_eval = np.expand_dims(ids_eval, axis=1)
        elif self._config.decoding == "beam_search":
            ids_eval = np.transpose(ids_eval, [0, 2, 1])

        for preds in ids_eval:
            for i, pred in enumerate(preds):
                p = truncate_end(pred, self._vocab.id_end)
                p = " ".join([self._vocab.id_to_tok[idx] for idx in p])
                hyps[i].append(p)

        return hyps

    def predict(self, img):
        preds = self.predict_batch([img])
        preds_ = []
        # extract only one element (no batch)
        for hyp in preds:
            preds_.append(hyp[0])

        return preds_
