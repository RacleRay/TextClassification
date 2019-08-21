import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell


class HANConfig(object):
    vocab_size = 5000
    embedding_size = 64
    classes = 10
    seq_length = 600         # 输入文件为一行为一篇文档，so取600没问题

    word_cell = GRUCell(64)
    sentence_cell = GRUCell(64)
    word_output_size = 64
    sentence_output_size = 64

    max_grad_norm = 5.0
    dropout_keep_prob = 0.8
    learning_rate = 1e-4

    batch_size = 128
    num_epochs = 10

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class HANModel(object):
    """Hierarchical Attention Networks"""
    def __init__(self, config):
        self.config = config

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # 输入为3维tensor，将每个doc以num_sentence x words of a sentence表示：
        # [num_document x num_sentence x num_word]
        self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')
        # 每个句子的长度：[num_document x num_sentence]
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')
        # 每个文档句子个数：[num_document]
        self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
        # [num_document x one hot dim]
        self.labels = tf.placeholder(shape=(None, None), dtype=tf.int32, name='labels')
        self.document_size, self.sentence_size, self.word_size = tf.unstack(tf.shape(self.inputs))
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.run()

    def run(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                                        [self.config.vocab_size, self.config.embedding_size],
                                        initializer=layers.xavier_initializer(),
                                        dtype=tf.float32)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

        with tf.variable_scope('word_encoder'):
            # inputs
            # 以一句话为单位输入
            word_level_inputs = tf.reshape(embedding_inputs,
                [self.document_size * self.sentence_size, self.word_size, self.config.embedding_size],
                name='word_level_inputs')
            word_encoder_length = tf.reshape(self.word_lengths,
                [self.document_size * self.sentence_size])

            word_encoder_output, _ = self.bidirectional_rnn(self.config.word_cell,
                                                       self.config.word_cell,
                                                       word_level_inputs,
                                                       word_encoder_length)
            # word attention
            word_level_output = self.context_attention(word_encoder_output,
                                                  output_size=self.config.word_output_size)
            word_level_output = layers.dropout(word_level_output,
                                               keep_prob=self.keep_prob)

        with tf.variable_scope('sentence_encoder'):
            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.config.word_output_size])

            sentence_encoder_output, _ = self.bidirectional_rnn(self.config.sentence_cell,
                                                                self.config.sentence_cell,
                                                                sentence_inputs,
                                                                self.sentence_lengths)
            sentence_level_output = self.context_attention(sentence_encoder_output,
                                                      output_size=self.config.sentence_output_size)
            sentence_level_output = layers.dropout(sentence_level_output,
                                                    keep_prob=self.keep_prob)

        with tf.variable_scope('classifier'):
            self.logits = tf.layers.dense(sentence_level_output,
                                          self.config.classes,
                                          name='logits')
            self.y_pred_cls = tf.argmax(self.logits, axis=-1)

        with tf.variable_scope('train'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy)

            # top_k = tf.nn.in_top_k(self.logits, self.labels, 1)
            # self.acc = tf.reduce_mean(tf.cast(top_k, tf.float32))
            correct_pred = tf.equal(tf.argmax(self.labels, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # 梯度裁剪
            trainables = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, trainables),
                self.config.max_grad_norm)
            opt = tf.train.AdamOptimizer(self.config.learning_rate)
            self.optim = opt.apply_gradients(zip(grads, trainables),
                                            name='train_op',
                                            global_step=self.global_step)


    @staticmethod
    def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths):
        """返回拼接的cell state和hidden state，以及每个每个step的输出"""
        with tf.variable_scope("birnn") as scope:
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs_embedded,
                sequence_length=input_lengths,
                dtype=tf.float32,
                swap_memory=True,
                scope=scope)

            outputs = tf.concat((fw_outputs, bw_outputs), 2)

            def concatenate_state(fw_state, bw_state):
                """拼接cell state， hidden state。multilayer同样适用"""
                if isinstance(fw_state, LSTMStateTuple):
                    state_c = tf.concat(
                        (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                    state_h = tf.concat(
                        (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                    state = LSTMStateTuple(c=state_c, h=state_h)
                    return state
                elif isinstance(fw_state, tf.Tensor):
                    state = tf.concat((fw_state, bw_state), 1,
                                    name='bidirectional_concat')
                    return state
                elif (isinstance(fw_state, tuple) and
                        isinstance(bw_state, tuple) and
                        len(fw_state) == len(bw_state)):
                    # multilayer
                    state = tuple(concatenate_state(fw, bw)
                                for fw, bw in zip(fw_state, bw_state))
                    return state

                else:
                    raise ValueError(
                        'unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state

    @staticmethod
    def context_attention(inputs, output_size,
                          initializer=layers.xavier_initializer(),
                          activation_fn=tf.tanh):
        """attention using learned context vector.
        context vector randomly initialized.

        args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
            outputs: Tensor of shape [batch_size, output_size].
        """
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

        with tf.variable_scope('attention'):
            context_vector = tf.get_variable(name='context_vector',
                                             shape=[output_size],
                                             initializer=initializer,
                                             dtype=tf.float32)
            # attention维度统一
            input_projection = tf.layers.dense(inputs,
                                               output_size,
                                               activation=activation_fn)
            vector_attn = tf.reduce_sum(input_projection * context_vector,
                                        axis=2,
                                        keep_dims=True)
            attn_weight = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = attn_weight * input_projection

            # 每个word representation进行加权求和，得到sentence representation
            outputs = tf.reduce_sum(weighted_projection, axis=1)

            return outputs
