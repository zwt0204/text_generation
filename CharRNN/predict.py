# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2020/1/10 12:45
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf
from RnnModel import RNN
from train import TextConverter
import logging
import numpy as np


class Predict:

    def __init__(self):
        self.graph = tf.Graph()
        self.learning_rate = 0.0001
        self.max_langth = 10
        self.model_path = 'model'
        self.is_training = False
        self.vocab_file = 'model\\converter.pkl'
        self.num_seqs, self.num_steps = 1, 1
        self.converter = TextConverter(filename=self.vocab_file)
        with self.graph.as_default():
            self.model = RNN(self.converter.vocab_size, num_steps=self.num_steps, num_seqs=self.num_seqs,
                             learning_rate=self.learning_rate, is_training=self.is_training)
            self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            logging.info("load success...")
        else:
            raise Exception("load failure...")

    def get_n_top(self, probs, vocab, n=5):
        p = np.squeeze(probs)
        # 1. 只选取概率最高的n个词，所以需要将其他位置的概率置为0
        # 2. 并重新计算n个词的取值概率，即归一化
        # 置0
        p[np.argsort(p)[:-n]] = 0
        # 归一化
        p = p / np.sum(p)
        # 在概率最大的前五个随机选择一个，保障多样性
        c = np.random.choice(vocab, 1, p=p)
        return c

    def predict(self, start):
        start = self.converter.text_to_arr(start)
        samples = [c for c in start]
        new_state = self.session.run(self.model.initial_state)
        # 初始化概率
        preds = np.ones((self.converter.vocab_size,))
        for c in start:
            # 输入单个字符
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.model.inputs: x,
                    self.model.keep_prob: self.model.keep_probs,
                    self.model.initial_state: new_state
                    }
            preds, new_state = self.session.run([self.model.prob, self.model.final_state], feed_dict=feed)

        c = self.get_n_top(preds, vocab=self.converter.vocab_size)
        samples.append(c)

        for i in range(self.max_langth):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.model.inputs: x,
                    self.model.keep_prob: self.model.keep_probs,
                    self.model.initial_state: new_state
                    }
            preds, new_state = self.session.run([self.model.prob, self.model.final_state], feed_dict=feed)
            c = self.get_n_top(preds, vocab=self.converter.vocab_size)
            samples.append(c)

        return self.converter.arr_to_text(np.array(samples))


if __name__ == '__main__':
    predict = Predict()
    result = predict.predict('独')
    print(result)
