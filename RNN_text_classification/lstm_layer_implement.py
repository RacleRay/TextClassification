#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def lstm_layer(name, inputs, hiddensize, initializer):
    """lstm layer实现"""
    inputshape = inputs.get_shape().as_list()
    if len(inputshape) == 2:
        inputs = tf.expand_dims(inputs, 0)

    inputshape = inputs.get_shape().as_list()
    assert len(inputs.get_shape().as_list()) == 3
    
    def _generate_params(x_size, h_size, bias_size):
        """generates parameters for pure lstm implementation."""
        # x输入权重矩阵
        x_w = tf.get_variable('x_weights', x_size)
        # hidden state传递权重矩阵
        h_w = tf.get_variable('h_weights', h_size)
        # 门运算前的偏置
        b = tf.get_variable('biases', bias_size,
                            initializer=tf.constant_initializer(0.0))
        return x_w, h_w, b

    with tf.variable_scope(name, initializer = initializer):
        with tf.variable_scope('inputs'):
            # 输入门的参数定义
            ix, ih, ib = _generate_params(
                x_size = [inputshape[-1], hiddensize],
                h_size = [hiddensize, hiddensize],
                bias_size = [1, hiddensize])

        with tf.variable_scope('outputs'):
            # 输出门的参数定义
            ox, oh, ob = _generate_params(
                x_size = [inputshape[-1], hiddensize],
                h_size = [hiddensize, hiddensize],
                bias_size = [1, hiddensize])

        with tf.variable_scope('forget'):
            # 遗忘门的参数定义
            fx, fh, fb = _generate_params(
                x_size = [inputshape[-1], hiddensize],
                h_size = [hiddensize, hiddensize],
                bias_size = [1, hiddensize])
            
        with tf.variable_scope('memory'):
            # 记忆单元参数定义
            cx, ch, cb = _generate_params(
                x_size = [inputshape[-1], hiddensize],
                h_size = [hiddensize, hiddensize],
                bias_size = [1, hiddensize])

        # 初始化cell state[0]，不需要训练
        cell_state = tf.Variable(
            tf.zeros([inputshape[0], hiddensize]),
            trainable = False
        )
        # 初始化hidden state[0]
        h = tf.Variable(
            tf.zeros([inputshape[0], hiddensize]),
            trainable = False
        )

        outputs_every_step = []
        for i in range(inputshape[1]):
            embed_input = embed_inputs[:, i, :]
            embed_input = tf.reshape(embed_input,
                                     [inputshape[0], inputshape[2]])
            # 输入：embed_input和h
            forget_gate = tf.sigmoid(
                tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
            # 输入：embed_input和h
            input_gate = tf.sigmoid(
                tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
            # 输入：embed_input和h
            input_info = tf.tanh(
                tf.matmul(embed_input, cx) + tf.matmul(embed_input, ch) + cb)
            # 输入：embed_input和h
            output_gate = tf.sigmoid(
                tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)

            # 更新cell state（element-wise） [batch, hiddensize]
            cell_state = cell_state * forget_gate + input_info * input_gate
            # 更新hidden state
            h = tf.tanh(cell_state) * output_gate

            outputs_every_step.append(h)
        last_output = outputs_every_step[-1]
    
    return outputs_every_step, last_output