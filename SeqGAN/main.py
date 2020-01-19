# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2020/1/17 14:19
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from train import Trainer


class Main:
    
    def __init__(self):
        self.g_pre_weights_path = "../data/save/generator_pre.hdf5"
        self.d_pre_weights_path = "../data/save/discriminator_pre.hdf5"
        self.g_weights_path = "../data/save/generator.pkl"
        self.d_weights_path = "../data/save/discriminator.hdf5"

        self.path_pos = "../data/train.txt"
        self.path_neg = "../data/save/generated_sentences.txt"

        self.g_test_path = "data.txt"
        self.batch_size = 32
        self.max_length = 25
        # Generator embedding size
        self.g_e = 64
        # Generator LSTM hidden size
        self.g_h = 64
        # Discriminator embedding and Highway network sizes
        self.d_e = 64
        # Discriminator LSTM hidden size
        self.d_h = 64

        # Number of Monte Calro Search
        self.n_sample = 16
        # Number of generated sentences
        self.generate_samples = 10

        # Pretraining parameters
        self.g_pre_epochs = 2
        self.d_pre_epochs = 1
        # [floats]
        self.g_lr = 1e-5

        # Discriminator dropout ratio
        self.d_dropout = 0.0
        self.d_lr = 1e-6

        # Pretraining parameters
        self.g_pre_lr = 1e-2
        self.d_pre_lr = 1e-4
        self.g_steps = 1
        self.d_steps = 5
        self.steps = 1

        # [lists]
        # filter sizes for CNNs
        self.d_filter_sizes = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20
        # num of filters for CNNs
        self.d_num_filters = 100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
        self.trainer = Trainer(self.batch_size,
                          self.max_length,
                          self.g_e,
                          self.g_h,
                          self.d_e,
                          self.d_h,
                          self.d_dropout,
                          path_pos=self.path_pos,
                          path_neg=self.path_neg,
                          g_lr=self.g_lr,
                          d_lr=self.d_lr,
                          n_sample=self.n_sample,
                          generate_samples=self.generate_samples)

    def PreTrain(self):
        self.trainer.pre_train(g_epochs=self.g_pre_epochs,
                          d_epochs=self.d_pre_epochs,
                          g_pre_path=self.g_pre_weights_path,
                          d_pre_path=self.d_pre_weights_path,
                          g_lr=self.g_pre_lr,
                          d_lr=self.d_pre_lr)

        # 加载权重
        self.trainer.load_pre_train(self.g_pre_weights_path, self.d_pre_weights_path)
        # 分配权重
        self.trainer.reflect_pre_train()

    def train(self):
        self.trainer.train(steps=self.steps,
                      g_steps=self.g_steps,
                      d_steps=self.d_steps,
                      head=10,
                      g_weights_path=self.g_weights_path,
                      d_weights_path=self.d_weights_path)

        self.trainer.save(self.g_weights_path, self.d_weights_path)

        self.trainer.load(self.g_weights_path, self.d_weights_path)

    def test(self):
        self.trainer.test()
        self.trainer.generate_txt(self.g_test_path, self.generate_samples)


if __name__ == '__main__':
    main = Main()
    main.PreTrain()
    main.train()