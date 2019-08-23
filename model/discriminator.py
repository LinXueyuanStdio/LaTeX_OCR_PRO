
import tensorflow as tf

from model.decoder import Decoder, embedding_initializer, get_embeddings
from model.encoder import Encoder
from model.components.BiLSTM_Attention import BiLSTM_Attention


class Discriminator(object):
    """判别器 Discriminator"""

    def __init__(self, config, n_tok):
        """这是判别器，啥配置也没有，莽就对了"""
        self._config = config
        self._n_tok = n_tok

    def __call__(self, fake_formula, formula, dropout):
        """
        N : N batchSize
        T : Tokens in vocab.txt
        Args:
            formula_length: (tf.placeholder), shape = (N) 最大公式长度
            fake_formula, formula: (tf.placeholder), shape = (N, T) 生成器的假公式 或 真实公式，这里需要自己判别
        Returns:

        """
        max_length_formula = self._config.max_length_formula
        dim_embeddings = self._config.attn_cell_config.get("dim_embeddings")
        dim_e = self._config.attn_cell_config["dim_e"]
        dim_o = self._config.attn_cell_config["dim_o"]
        num_units = self._config.attn_cell_config["num_units"]
        l2RegLambda = 0.0
        with tf.variable_scope("Discriminator"):
            embedding_table = tf.get_variable("embedding_table", shape=[self._n_tok, dim_embeddings],
                                              dtype=tf.float64, initializer=embedding_initializer())
            start_token = tf.get_variable("start_token", shape=[dim_embeddings],
                                          dtype=tf.float64, initializer=embedding_initializer())
            batch_size = tf.shape(formula)[0]
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            with tf.variable_scope("BiLSTM_Attention", reuse=False):
                embeddings = get_embeddings(formula, embedding_table, dim_embeddings,
                                            start_token, batch_size)  # (N, T, dim_embedding)
                lstm = BiLSTM_Attention(embeddings,
                                        batch_size, max_length_formula, dim_embeddings,
                                        hiddenSizes=[dim_e, dim_o])
                D_logit_real, D_predictoin_real, l2Loss_real = lstm(dropout)
                D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_real,
                    labels=tf.ones_like(D_logit_real))) + l2RegLambda * l2Loss_real
            with tf.variable_scope("BiLSTM_Attention", reuse=True):
                fake_formula = fake_formula[:, :, 0]
                fake_embeddings = get_embeddings(fake_formula, embedding_table, dim_embeddings,
                                                 start_token, batch_size)  # (N, T, dim_embedding)
                lstm = BiLSTM_Attention(fake_embeddings,
                                        batch_size, max_length_formula, dim_embeddings,
                                        hiddenSizes=[dim_e, dim_o])
                D_logit_fake, D_predictoin_fake, l2Loss_fake = lstm(dropout)
                D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_fake,
                    labels=tf.zeros_like(D_logit_fake))) + l2RegLambda * l2Loss_real
            self.D_loss = D_loss_real + D_loss_fake
        return self.D_loss
