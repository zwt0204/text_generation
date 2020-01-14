# -*- encoding: utf-8 -*-
"""
@File    : model.py
@Time    : 2020/1/13 10:32
@Author  : zwt
@git   : https://github.com/jinfagang/tensorflow_poems
@Software: PyCharm
"""
import tensorflow as tf


class RNN:

    def __init__(self, model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
        self.model = model
        self.input_data = input_data
        self.output_data = output_data
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.end_points = {}
        self.create_declare()
        self.create_embedding()
        self.create_mdoel()

    def create_declare(self):
        self.weights = tf.Variable(tf.truncated_normal([self.rnn_size, self.vocab_size + 1]))
        self.bias = tf.Variable(tf.zeros(shape=[self.vocab_size + 1]))

    def create_embedding(self):
        with tf.device("/cpu:0"):
            # random_uniform均匀分布
            embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
                [self.vocab_size + 1, self.rnn_size], -1.0, 1.0))
            self.inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    def create_mdoel(self):
        if self.model == 'rnn':
            self.cell_fun = tf.contrib.rnn.BasicRNNCell
        elif self.model == 'gru':
            self.cell_fun = tf.contrib.rnn.GRUCell
        elif self.model == 'lstm':
            self.cell_fun = tf.contrib.rnn.BasicLSTMCell
        cell = self.cell_fun(self.rnn_size, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        if self.output_data is not None:
            initial_state = cell.zero_state(self.batch_size, tf.float32)
        else:
            initial_state = cell.zero_state(1, tf.float32)

        # [batch_size, ?, rnn_size] = [64, ?, 128]
        outputs, last_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, self.rnn_size])
        logits = tf.nn.bias_add(tf.matmul(output, self.weights), bias=self.bias)
        if self.output_data is not None:
            # output_data must be one-hot encode
            labels = tf.one_hot(tf.reshape(self.output_data, [-1]), depth=self.vocab_size + 1)
            # should be [?, vocab_size+1]

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # loss shape should be [?, vocab_size+1]
            total_loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

            self.end_points['initial_state'] = initial_state
            self.end_points['output'] = output
            self.end_points['train_op'] = train_op
            self.end_points['total_loss'] = total_loss
            self.end_points['loss'] = loss
            self.end_points['last_state'] = last_state
        else:
            prediction = tf.nn.softmax(logits)

            self.end_points['initial_state'] = initial_state
            self.end_points['last_state'] = last_state
            self.end_points['prediction'] = prediction