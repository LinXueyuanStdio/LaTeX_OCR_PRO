import time
import math

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import tensorflow as tf
import numpy as np
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getWH(img_w, img_h):
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w - 2), np.ceil(img_h - 2)
    return int(img_w), int(img_h)


class EncoderCNN(nn.Module):
    def __init__(self, config, training=False):
        super(EncoderCNN, self).__init__()
        self._config = config
        self.cnn = self.getCNN(self._config.encoder_cnn)

    def getCNN(self, cnn_name):
        if cnn_name == "vanilla":
            return nn.Sequential(
                # conv + max pool -> /2
                # 64 个 3*3 filters, strike = (1, 1), output_img.shape = ceil(L/S) = ceil(input/strike) = (H, W)
                nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # conv + max pool -> /2
                nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # regular conv -> id
                nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                # conv
                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )
        elif cnn_name == "cnn":
            return nn.Sequential(
                # conv + max pool -> /2
                # 64 个 3*3 filters, strike = (1, 1), output_img.shape = ceil(L/S) = ceil(input/strike) = (H, W)
                nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # conv + max pool -> /2
                nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # regular conv -> id
                nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=(2, 4), stride=2, padding=1),
                nn.ReLU(),

                # conv
                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )

    def forward(self, img):
        """
        Args:
            img: [batch, channel, W, H]
        return:
            out: [batch, W/2/2/2-2, H/2/2/2-2, 512]
        """
        out = self.cnn(img)
        if self._config.positional_embeddings:
            # positional embeddings
            out = self.add_timing_signal_nd_torch(out)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.cnn.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.cnn.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def add_timing_signal_nd_torch(self, x, min_timescale=1.0, max_timescale=1.0e4):
        """嵌入位置信息（positional）
        在Tensor中添加一堆不同频率的正弦曲线。即给输入张量的每个 channels 在对应 positional 维度中加上不同频率和相位的正弦曲线。

        这可以让注意力层学习到绝对和相对位置。
        将信号添加到 query 和 memory 输入当成注意力。
        使用相对位置是可能的，因为 sin(a + b) 和 cos(a + b) 可以用 b，sin(a)和cos(a) 表示。也就是换个基。

        x 是有 n 个 “positional” 维的张量，例如序列是一维，一个 positional 维；图像是二维，两个 positional 维

        我们使用从 min_timescale 开始到 max_timescale 结束的 timescales 序列。不同 timescales 的数量等于 channels//（n * 2）。
        对于每个 timescales，我们生成两个正弦信号 sin(timestep/timescale) 和 cos(timestep/timescale)。
        在 channels 维上将这些正弦曲线连接起来。
        """
        static_shape = x.shape  # [20, 512, 14, 14]
        num_dims = len(static_shape) - 2  # 2
        channels = static_shape[1]  # 512
        num_timescales = channels // (num_dims * 2)  # 128
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1)
        )  # 0.1
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * (-log_timescale_increment))  # len == 128
        for dim in range(num_dims):  # dim == 0; 1
            length = static_shape[dim + 2]  # 14
            position = torch.arange(length).float()  # len == 14
            # inv = [128, 1]， pos = [1, 14], scaled_time = [128, 14]
            scaled_time = inv_timescales.unsqueeze(1) * position.unsqueeze(0)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=0)  # [256， 14]

            prepad = dim * 2 * num_timescales  # 0; 256
            postpad = channels - (dim + 1) * 2 * num_timescales  # 256; 0

            signal = F.pad(signal, (0, 0, prepad, postpad))  # [512, 14]

            signal = signal.unsqueeze(0)
            for _ in range(dim):
                signal = signal.unsqueeze(2)  # [512, 14]
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-1)
            x += signal  # [1, 512, 14, 1]; [1, 512, 1, 14]
        return x


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class DecoderRNN(nn.Module):
    def __init__(self, config, n_tok, id_end):
        super(DecoderRNN, self).__init__()
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == "greedy" else config.beam_size
        self.dim_embeddings = self._config.attn_cell_config.get("dim_embeddings")

        self.embed = nn.Embedding(self._n_tok, self.dim_embeddings)
        # self.lstm = nn.LSTM(self.dim_embeddings, hidden_size, num_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, vocab_size)
        # self.max_seg_length = max_seq_length

    def forward(self, img, formula, lengths):
        """Decodes an image into a sequence of token

        Args:
            img: encoded image, shape = (N, H, W, C) (N, H/2/2/2-2, W/2/2/2-2, 512)
            formula: shape = (N, T)
            lengths: length of formula, == n_tok

        Returns:
            pred_train: (tf.Tensor), shape = (?, ?, vocab_size) logits of each class
            pret_test: (structure)
                - pred.test.logits, same as pred_train
                - pred.test.ids, shape = (?, config.max_length_formula)

        """
        embeddings = self.embed(formula)  # (N, T, dim_embedding)
        embeddings = torch.cat((formula.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class Img2Seq(nn.Module):
    def __init__(self, config, vocab):
        super(Img2Seq, self).__init__()
        self._config = config
        self.encoder = EncoderCNN(config)
        # self.decoder = DecoderRNN(config, vocab.n_tok, vocab.id_end)
        self._vocab = vocab

    def forward(self):
        pass

    # def parameters(self):
    #     return list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())

"""
0.6 偏微分方程 1. 有限元法 2. 有限差分法
0.4 常微分方程 今天开始算考试的内容
"""