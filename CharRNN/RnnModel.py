# -*- encoding: utf-8 -*-
"""
@File    : RnnModel.py
@Time    : 2020/1/10 9:44
@Author  : zwt
@git   : https://github.com/fennuDetudou/text_generate/blob/master/model.py
@Software: PyCharm
"""
import tensorflow as tf


class RNN:

    def __init__(self, num_classes, num_seqs=32, num_steps=50, lstm_size=128, num_layers=2,
                 embedding_size=64, is_training=False, learning_rate=0.001):
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        if self.is_training is True:
            self.keep_probs = 0.5
        else:
            self.keep_probs = 1.0

        with tf.name_scope('inputs'):
            # 输入要转化为one_hot向量，要用int而不是float
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps))
            self.labels = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps))
            self.keep_prob = tf.placeholder(tf.float32, shape=None)

        self.create_embedding()
        self.create_model()
        self.create_loss()

    def create_embedding(self):
        with tf.name_scope('declare'):
            # tf embedding层
            # 1. 要在 TensorFlow 中创建 embeddings，我们首先将文本拆分成单词，然后为词汇表中的每个单词分配一个整数
            # 2. 利用整数组成的向量训练embedding层，输出shape为vocabulary_size,embedding_size
            # 3. 利用embedding_lookup(embedding层，整数向量)获得inputs的分布式表示
            self.embedding = tf.get_variable('embeddings', shape=(self.num_classes, self.embedding_size))
            self.w = tf.get_variable('softmax_w', shape=[self.lstm_size, self.num_classes], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer)
            self.b = tf.get_variable('softmax_b', self.num_classes, dtype=tf.float32)

    def cell(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell, self.keep_prob)
        return drop

    def create_model(self):
        self.lstm_inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)
        with tf.name_scope('rnn'):
            multi_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell() for _ in range(self.num_layers)])
            self.initial_state = multi_cell.zero_state(self.num_seqs, tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(multi_cell, self.lstm_inputs, initial_state=self.initial_state)
            self.outputs = tf.concat(outputs, 1)
            x = tf.reshape(self.outputs, [-1, self.lstm_size])

            self.logits = tf.matmul(x, self.w) + self.b
            self.prob = tf.nn.softmax(self.logits)

    def create_loss(self):
        with tf.name_scope('loss'):
            # softmax标签处理
            one_hot_labels = tf.one_hot(self.labels, self.num_classes)
            labels = tf.reshape(one_hot_labels, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # 返回的是正常计算的梯度
            gradient_vars = optimizer.compute_gradients(self.loss)
            cropped_gradient = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradient_vars]
            self.train_op = optimizer.apply_gradients(cropped_gradient)
