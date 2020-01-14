# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2020/1/13 10:53
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import RNN
import tensorflow as tf
import collections
import numpy as np
import json


class Predict:

    def __init__(self):
        self.batch_size = 1
        self.learning_rate = 0.0001
        self.max_words = 24
        self.model_dir = '../model'
        self.file_path = '../data/train.txt'
        self.start_token = 'B'
        self.end_token = 'E'
        _, self.word_int_map, self.vocabularies = self.process_poems(self.file_path)
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])

        self.model = RNN(model='lstm', input_data=self.input_data, output_data=None, vocab_size=len(
            self.vocabularies), rnn_size=128, num_layers=2, batch_size=self.batch_size, learning_rate=self.learning_rate)

    def process_poems(self, file_name):
        start_token = 'B'
        end_token = 'E'
        # poems -> list of numbers
        poems = []
        with open(file_name, "r", encoding='utf-8', ) as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                    content = data['text']
                    # title, content = line.strip().split(':')
                    content = content.replace(' ', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 79:
                        continue
                    content = start_token + content + end_token
                    poems.append(content)
                except ValueError as e:
                    pass

        all_words = [word for poem in poems for word in poem]
        counter = collections.Counter(all_words)
        words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)

        words.append(' ')
        L = len(words)
        word_int_map = dict(zip(words, range(L)))
        poems_vector = [list(map(lambda word: word_int_map.get(word, L), poem)) for poem in poems]
        return poems_vector, word_int_map, words

    def to_word(self, predict, vocabs):
        predict = predict[0]
        predict /= np.sum(predict)
        sample = np.random.choice(np.arange(len(predict)), p=predict)
        if sample > len(vocabs):
            return vocabs[-1]
        else:
            return vocabs[sample]

    def predict(self, begin_word):
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, checkpoint)
            x = np.array([list(map(self.word_int_map.get, self.start_token))])
            [predict, last_state] = sess.run([self.model.end_points['prediction'], self.model.end_points['last_state']],
                                             feed_dict={self.input_data: x})
            word = begin_word or self.to_word(predict, self.vocabularies)
            poem_ = ''
            i = 0
            while word != self.end_token:
                poem_ += word
                i += 1
                if i > self.max_words:
                    break
                x = np.array([[self.word_int_map[word]]])
                [predict, last_state] = sess.run([self.model.end_points['prediction'], self.model.end_points['last_state']],
                                                 feed_dict={self.input_data: x, self.model.end_points['initial_state']: last_state})
                word = self.to_word(predict, self.vocabularies)
            poem_sentences = poem_.split('。')
            for s in poem_sentences:
                if s != '' and len(s) > 10:
                    print(s + '。')


if __name__ == '__main__':
    predict = Predict()
    predict.predict('你')