# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/1/10 10:49
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
import pickle
import codecs
import os
from RnnModel import RNN
import copy
import tensorflow as tf


class Train:

    def __init__(self):
        self.input_file = 'D:\mygit\\text_generation\poetry'
        self.max_vocab = 5000
        self.model_path = 'model'
        self.converter = self.load_dict()
        self.vocab_size = self.converter.vocab_size
        self.is_training = True
        self.learning_rate = 0.0001
        self.model = RNN(self.vocab_size, learning_rate=self.learning_rate, is_training=self.is_training)
        self.saver = tf.train.Saver()

    def load_dict(self):
        with codecs.open(self.input_file, encoding='utf-8') as f:
            # 将整个文件读取为字符串
            self.text = f.read()
        converter = TextConverter(self.text, self.max_vocab)
        converter.save_to_file(os.path.join(self.model_path, 'converter.pkl'))
        return converter

    def train(self, epochs=100000):
        arr = self.converter.text_to_arr(self.text)
        batch_generator = self.batch_generator(arr, self.model.num_seqs, self.model.num_steps)
        initer = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(initer)
            new_state = session.run(self.model.initial_state)
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None and ckpt.model_checkpoint_path:
                self.saver.restore(session, ckpt.model_checkpoint_path)
            step = 0
            for x, y in batch_generator:
                step += 1
                feed = {
                    self.model.inputs: x,
                    self.model.labels: y,
                    self.model.keep_prob: self.model.keep_probs,
                    self.model.initial_state: new_state
                }
                batch_loss, new_state, _ = session.run([self.model.loss, self.model.final_state, self.model.train_op],
                                                          feed_dict=feed)
                if step % 100 == 0:
                    print('step: {}/{}... '.format(step, epochs),
                          'loss: {:.4f}... '.format(batch_loss))

                if step > epochs:
                    break
                self.saver.save(session, os.path.join(self.model_path, 'model'), step)

    def batch_generator(self, arr, num_seqs, num_steps):
        '''
        将一段文本数据转换为(num_seqs,len(arr)/num_seqs)
        eg:12345678 num_seqs=2
        1234
        5678
        输出数据shape（num_seqs,num_steps）
        '''
        arr = copy.copy(arr)
        batch_size = num_seqs * num_steps
        n_batches = int(len(arr) / batch_size)
        arr = arr[:batch_size * n_batches]
        arr = arr.reshape((num_seqs, -1))
        while True:
            np.random.shuffle(arr)
            for n in range(0, arr.shape[1], num_steps):
                x = arr[:, n:n + num_steps]
                y = np.zeros_like(x)
                # x的后一个字符作为y的字符
                # y[:, :-1]表示第一维度全部，第二维度取到倒数第二个
                # y[:, -1]表示第一维度全部，第二维度最后一个
                # x[:, 1:]表示第一维度全部，第二维度第二个开始到最后一个
                # x[:, 0]表示第一维度全部，第二维度的第一个
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
                yield x, y


# 文本数据向量化
class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print(len(vocab))
            # 找出最大的max_vocab个单词
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab
        # 字转id以及id转字
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        index = int(index)
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        # 字转id
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        # id 转字
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)


if __name__ == '__main__':
    train = Train()
    train.train()