# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2020/1/13 10:42
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model import RNN
import tensorflow as tf
import collections
import numpy as np
import os
import json


class Train:

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.model_dir = '../model'
        self.file_path = '../data/train.txt'
        self.model_prefix = 'poem'
        self.epochs = 10
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
        self.output_targets = tf.placeholder(tf.int32, [self.batch_size, None])
        self.poems_vector, self.word_to_int, self.vocabularies = self.process_poems(self.file_path)
        self.model = RNN(model='lstm', input_data=self.input_data, output_data=self.output_targets, vocab_size=len(
        self.vocabularies), rnn_size=128, num_layers=2, batch_size=self.batch_size, learning_rate=self.learning_rate)
        self.saver = tf.train.Saver(tf.global_variables())

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

    def generate_batch(self, batch_size, poems_vec, word_to_int):
        n_chunk = len(poems_vec) // batch_size
        x_batches = []
        y_batches = []
        for i in range(n_chunk):
            start_index = i * batch_size
            end_index = start_index + batch_size

            batches = poems_vec[start_index:end_index]
            length = max(map(len, batches))
            # np.full(shape, fill_value)可以生成一个元素为fill_value，形状为shape的array
            x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
            for row, batch in enumerate(batches):
                x_data[row, :len(batch)] = batch
            y_data = np.copy(x_data)
            y_data[:, :-1] = x_data[:, 1:]
            """
            x_data             y_data
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
            x_batches.append(x_data)
            y_batches.append(y_data)
        return x_batches, y_batches

    def train(self):
        batches_inputs, batches_outputs = self.generate_batch(self.batch_size, self.poems_vector, self.word_to_int)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            start_epoch = 0
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if checkpoint:
                self.saver.restore(sess, checkpoint)
                print("## restore from the checkpoint {0}".format(checkpoint))
                start_epoch += int(checkpoint.split('-')[-1])
            print('## start training...')
            try:
                n_chunk = len(self.poems_vector) // self.batch_size
                for epoch in range(start_epoch, self.epochs):
                    n = 0
                    for batch in range(n_chunk):
                        loss, _, _ = sess.run([
                            self.model.end_points['total_loss'],
                            self.model.end_points['last_state'],
                            self.model.end_points['train_op']
                        ], feed_dict={self.input_data: batches_inputs[n], self.output_targets: batches_outputs[n]})
                        n += 1
                        print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                    if epoch % 1 == 0:
                        self.saver.save(sess, os.path.join(self.model_dir, self.model_prefix), global_step=epoch)
            except KeyboardInterrupt:
                print('## Interrupt manually, try saving checkpoint for now...')
                self.saver.save(sess, os.path.join(self.model_dir, self.model_prefix), global_step=epoch)
                print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


if __name__ == '__main__':
    train = Train()
    train.train()