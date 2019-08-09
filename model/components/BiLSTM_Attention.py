import tensorflow as tf

from model.decoder import embedding_initializer, get_embeddings


class BiLSTM_Attention(object):
    """
    双向 LSTM + Attention
    """

    def __init__(self, embeddings,
                 batch_size, max_formula_length, dim_embeddings,
                 hiddenSizes=[256, 128]):
        """
        T : Tokens index in vocab.txt
        Args:
            embeddings: (tf.float32), shape = (batch_size, tokens_count, dim_embeddings) 生成器的假公式 或 真实公式，这里需要训练才能判别
            batch_size         == tf.shape(embeddings)[0],  (tf.Shape(), unknown, but we can kown it when there is data in graph)
            max_formula_length == config.max_formula_length (int)
            dim_embeddings     == config.dim_embeddings     (int)
            hiddenSizes, 双向 LSTM
        """
        tokens_count = tf.shape(embeddings)[1] # (tf.Shape(), unknown, but we can kown it when there is data in graph)
        # 这个 tokens_count 导致我们之后的计算全部用计算图的形式构造 shape=(batch_size, max_formula_length, dim_embeddings) 的 embedding
        # 因为 双向 LSTM 上加 Attention，要求输入是定长的，需要把 embeddings 转化为 定长 的 Tensor，超过则截断，不足则补零
        print(embeddings.shape)
        print('shape : ', [batch_size, max_formula_length, dim_embeddings])

        def tokenscount_less_than_maxlength():
            b = tf.zeros([batch_size, max_formula_length, dim_embeddings], dtype=tf.float32)
            d = tf.slice(embeddings, [0, 0, 0], [batch_size, tokens_count, dim_embeddings])
            # d 相当于 embeddings[0:batch_size, 0:tokens_count, 0:dim_embeddings]，只是写成计算图的形式
            e = tf.slice(b, [0, 0, 0], [batch_size, max_formula_length-tokens_count, dim_embeddings])
            # e 相当于 b[0:batch_size, 0:max_formula_length-tokens_count, 0:dim_embeddings]，只是写成计算图的形式
            new_embeddings = tf.reshape(tf.concat([d, e], 1), [-1, max_formula_length, dim_embeddings])
            return new_embeddings

        def tokenscount_greater_than_maxlength():
            return tf.slice(embeddings, [0, 0, 0], [batch_size, max_formula_length, dim_embeddings])

        new_embeddings = tf.cond(tokens_count < max_formula_length,
                                 tokenscount_less_than_maxlength,
                                 tokenscount_greater_than_maxlength)
        # 上面这句相当于以下代码
        # if (tokens_count < max_formula_length):
        #     new_embeddings = tokenscount_less_than_maxlength()
        # else:
        #     new_embeddings = tokenscount_greater_than_maxlength()
        #

        self._embeddings = new_embeddings
        self._hiddenSizes = hiddenSizes
        self._max_formula_length = max_formula_length

    def __call__(self, dropout):
        """
        N : N batchSize
        T : Tokens index in vocab.txt
        Args:
            formula: (tf.placeholder), shape = (N, T) 生成器的假公式 或 真实公式，这里需要自己判别
        Returns:
            logits: 用于计算模型损失
            predictions
            l2Loss: 计算二元交叉熵损失
        可以取 l2RegLambda = 0.0，则
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputY)
            loss = tf.reduce_mean(losses) + l2RegLambda * l2Loss
        """
        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(self._hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,
                                                                                       state_is_tuple=True),
                                                               output_keep_prob=dropout)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,
                                                                                       state_is_tuple=True),
                                                               output_keep_prob=dropout)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    print("bi-lstm" + str(idx), 'hiddenSize', hiddenSize)
                    print("bi-lstm" + str(idx), 'self._embeddings', self._embeddings.shape)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self._embeddings,
                                                                                   dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self._embeddings = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self._embeddings, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H, dropout)
            outputSize = self._hiddenSizes[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW", shape=[outputSize, 1],
                                      initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")

        return self.logits, self.predictions, l2Loss

    def attention(self, H, dropout):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self._hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        print("H", H.shape)
        print("newM", newM.shape)
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self._max_formula_length])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self._max_formula_length, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, dropout)

        return output
