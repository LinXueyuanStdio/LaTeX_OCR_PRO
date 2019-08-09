
import tensorflow as tf

from model.decoder import Decoder
from model.encoder import Encoder


class Generator(object):
    """生成器 Generator，实际上是识别器 -- 将 Encoder 和 Decoder 组合形成的公式生成器 -- 公式将传给判别器"""

    def __init__(self, config, n_tok, id_end):
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, n_tok, id_end)

    def __call__(self, img, formula, dropout):
        """将 Encoder 和 Decoder 组合，形成 公式生成器
        Args:
            for Encoder:
            img: batch of img, shape = (?, height, width, channels), of type tf.uint8
            tf.uint8 因为 2^8 = 256，所以元素值区间 [0, 255]，线性压缩到 [-1, 1] 上就是 img = (img - 128) / 128

            for Decoder:
            formula: (tf.placeholder), shape = (N, T)
        Returns:
            Decoder's returns
        """
        with tf.variable_scope("Generator"):
            encoded_img = self.encoder(img, dropout)
            train, test = self.decoder(encoded_img, formula, dropout)
        return train, test
