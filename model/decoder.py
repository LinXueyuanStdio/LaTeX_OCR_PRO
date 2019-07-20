import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell, LSTMCell


from .components.dynamic_decode import dynamic_decode
from .components.attention_mechanism import AttentionMechanism
from .components.attention_cell import AttentionCell
from .components.greedy_decoder_cell import GreedyDecoderCell
from .components.beam_search_decoder_cell import BeamSearchDecoderCell


class Decoder(object):
    """Implements this paper https://arxiv.org/pdf/1609.04938.pdf"""

    def __init__(self, config, n_tok, id_end):
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == "greedy" else config.beam_size

    def __call__(self, img, formula, dropout):
        """Decodes an image into a sequence of token

        Args:
            img: encoded image (tf.Tensor) shape = (N, H, W, C) (N, H/2/2/2-2, W/2/2/2-2, 512)
            formula: (tf.placeholder), shape = (N, T)

        Returns:
            pred_train: (tf.Tensor), shape = (?, ?, vocab_size) logits of each class
            pret_test: (structure)
                - pred.test.logits, same as pred_train
                - pred.test.ids, shape = (?, config.max_length_formula) 主要用这个，id 直接就是 token 的 id 了

        """
        dim_embeddings = self._config.attn_cell_config.get("dim_embeddings")
        dim_e = self._config.attn_cell_config["dim_e"]
        num_units = self._config.attn_cell_config["num_units"]
        with tf.variable_scope("Decoder"):
            embedding_table = tf.get_variable("embedding_table", shape=[self._n_tok, dim_embeddings],
                                            dtype=tf.float32, initializer=embedding_initializer())

            start_token = tf.get_variable("start_token", shape=[dim_embeddings],
                                        dtype=tf.float32, initializer=embedding_initializer())

            batch_size = tf.shape(img)[0]
            # training
            with tf.variable_scope("AttentionCell", reuse=False):
                embeddings = get_embeddings(formula, embedding_table, dim_embeddings,
                                            start_token, batch_size)  # (N, T, dim_embedding)
                attn_meca = AttentionMechanism(img, dim_e)
                recu_cell = LSTMCell(num_units)
                attn_cell = AttentionCell(recu_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)

                train_outputs, _ = tf.nn.dynamic_rnn(attn_cell, embeddings, initial_state=attn_cell.initial_state())

            # decoding
            with tf.variable_scope("AttentionCell", reuse=True):
                attn_meca = AttentionMechanism(img, dim_e, tiles=self._tiles)
                recu_cell = LSTMCell(num_units, reuse=True)
                attn_cell = AttentionCell(recu_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)
                if self._config.decoding == "greedy":
                    decoder_cell = GreedyDecoderCell(embedding_table, attn_cell, batch_size, start_token, self._id_end)
                elif self._config.decoding == "beam_search":
                    decoder_cell = BeamSearchDecoderCell(embedding_table, attn_cell, batch_size, start_token, self._id_end,
                                                        self._config.beam_size, self._config.div_gamma, self._config.div_prob)

                test_outputs, _ = dynamic_decode(decoder_cell, self._config.max_length_formula+1)

        return train_outputs, test_outputs


def get_embeddings(formula, embedding_table, dim, start_token, batch_size):
    """Returns the embedding of the n-1 first elements in the formula concat
    with the start token

    Args:
        formula: (tf.placeholder) tf.uint32 shape = (N, T)
        embedding_table: tf.Variable (matrix) shape=[T, dim]
        dim: (int) dimension of embeddings
        start_token: tf.Variable shape=[dim]
        batch_size: tf variable extracted from placeholder

    Returns:
        embeddings_train: tensor

    """
    formula_ = tf.nn.embedding_lookup(embedding_table, formula)  # (N, T, dim_embedding)
    start_token_ = tf.reshape(start_token, [1, 1, dim])
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])  # (N, 1, dim_embedding)
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)  # (N, T, dim_embedding)

    return embeddings


def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        embedding_table = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        embedding_table = tf.nn.l2_normalize(embedding_table, -1)
        return embedding_table

    return _initializer
