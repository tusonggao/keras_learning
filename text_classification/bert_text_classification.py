# https://www.cnblogs.com/dogecheng/p/11617940.html
# https://www.cnblogs.com/dogecheng/p/11824494.html
# https://github.com/bojone/bert_in_keras/blob/master/sentiment.py
from __future__ import print_function, division, with_statement
import os
import sys
import time
import numpy as np
import pandas as pd

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary, Tokenizer
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

print('prog starts here!')

# 超参数
maxlen = 100
batch_size = 16
droup_out_rate = 0.5
learning_rate = 1e-5
epochs = 15

# 预训练模型目录
bert_path_prefix = '/home/ubuntu/ftpfile/chinese_wwm_ext_L-12_H-768_A-12/'
config_path = bert_path_prefix + "bert_config.json"
checkpoint_path = bert_path_prefix + "bert_model.ckpt"
dict_path = bert_path_prefix + "vocab.txt"

# 读取数据
data_path_prefix = './'
train_df = pd.read_excel(data_path_prefix + './train_new.csv')
test_df = pd.read_excel(data_path_prefix + './test_new.csv')

train_data = [(text, target) for text, target in zip(train_df.question_text.values, train_df.target.values)]
test_data = [(text, target) for text, target in zip(test_df.question_text.values, test_df.target.values)]

random.seed(2018)
random.shuffle(train_data)
train_num = int(0.9*len(data))
train_data = train_data[:train_num]
val_data = train_data[:int(0.9*len(

print('neg.shape is ', neg.shape, 'pos.shape is', pos.shape)

# 构建训练数据
data = []

# 读取字典
token_dict = load_vocabulary(dict_path)
# 建立分词器
tokenizer = Tokenizer(token_dict)

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

print('prog get here 222')

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# trainable设置True对Bert进行微调
# 默认不对Bert模型进行调参
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True)

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
x = Dropout(droup_out_rate)(x)
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)
model.summary()

print('prog get here 333')

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=epochs,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)

print('prog ends here!')


