import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import time
import numpy as np
import tensorflow as tf
import lightgbm as lgb

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def batcher(X_data, y_data, batch_size=-1, random_seed=None):
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    if random_seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(random_seed)

    rnd_idx = np.random.permutation(len(X_data))
    # print('rnd_idx[:10] is', rnd_idx[:10])
    n_batches = len(X_data) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X_data[batch_idx], y_data[batch_idx]
        yield X_batch, y_batch


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def larger_model():
    model = Sequential()
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model():
    model = Sequential()
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal'))
    model.add(Dense(7, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_1_old():
    model = Sequential()
    model.add(Dense(61, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal'))
    model.add(Dense(7, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_1_new():
    model = Sequential()
    model.add(Dense(61, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_2_old():
    model = Sequential()
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(77, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal'))
    model.add(Dense(11, kernel_initializer='normal'))
    model.add(Dense(7, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_2_new():
    model = Sequential()
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(77, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def largest_model_3_new():
    model = Sequential()
    model.add(Dense(131, kernel_initializer='normal', activation='relu'))
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(77, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def largest_model_4_new():
    model = Sequential()
    model.add(Dense(211, kernel_initializer='normal', activation='relu'))
    model.add(Dense(133, kernel_initializer='normal', activation='relu'))
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(77, kernel_initializer='normal', activation='relu'))
    model.add(Dense(57, kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dense(27, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def largest_model_5_new():
    model = Sequential()
    model.add(Dense(311, kernel_initializer='normal', activation='relu'))
    model.add(Dense(177, kernel_initializer='normal', activation='relu'))
    model.add(Dense(133, kernel_initializer='normal', activation='relu'))
    model.add(Dense(111, kernel_initializer='normal', activation='relu'))
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_6_new():
    model = Sequential()
    model.add(Dense(311, kernel_initializer='normal', activation='relu'))
    model.add(Dense(177, kernel_initializer='normal', activation='relu'))
    model.add(Dense(133, kernel_initializer='normal', activation='relu'))
    model.add(Dense(111, kernel_initializer='normal', activation='relu'))
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_6_new_bn():
    model = Sequential()
    model.add(Dense(311, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(177, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(133, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(111, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(91, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(51, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def largest_model_7_new_bn():
    model = Sequential()
    model.add(Dense(311, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(177, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(133, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def wider_model():
    model = Sequential()
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def regression_keras(X_train, y_train, X_test, y_test):
    X_merged = np.r_[X_train, X_test]
    X_merged = StandardScaler().fit_transform(X_merged)      # 标准转化
    X_merged = np.c_[np.ones((len(X_merged), 1)), X_merged]  # 增加一列1，用于学习bias数值
    # X_merged = StandardScaler().fit_transform(X_merged)  # 标准转化
    X_train = X_merged[:len(X_train)]
    X_test = X_merged[len(X_train):]
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # model = baseline_model()  # final test_loss is  0.4125122811058079
    # model = wider_model()  # final test_loss is  0.393576544692658
    # model = larger_model()  # final test_loss is  0.37739271818821435
    # model = largest_model()  # final test_loss is  0.3535420367884081
    # model = largest_model_1_old()  # final test_loss is  0.32797491369475384
    # model = largest_model_1_new()  # final test_loss is  0.3587214615224868  final test_loss is  0.3204049845382533
    # model = largest_model_2_old()  # final test_loss is  0.3283110881267592
    # model = largest_model_2_new()  # final test_loss is  0.33472770013550457  final test_loss is  0.3176132480226438
    # model = largest_model_3_new()  # final test_loss is  0.31482196751337027   final test_loss is  0.3108409709400601
    # model = largest_model_4_new()  # final test_loss is  0.29989305565985597  final test_loss is  0.32538971955616036  final test_loss is  0.29629954988334223
    # model = largest_model_5_new()  # final test_loss is  0.2898472023102664   final test_loss is  0.29703074012153835  final test_loss is  0.2863174539627339
    # model = largest_model_6_new()  # final test_loss is  0.3271503668923403  final test_loss is  0.2984573458169782  final test_loss is  0.34617327226840866
    # model = largest_model_6_new_bn()  # final test_loss is  0.3331636275893958  final test_loss is  0.33187634780887487  final test_loss is  0.41732066673363827
    model = largest_model_7_new_bn()  # final test_loss is  0.3102793127529381

    total_iter_num = 0
    max_epochs = 10
    batch_size = 64
    for epoch in range(max_epochs):
        print('current epoch is ', epoch)
        epoch_start_t = time.time()
        for batch_X, batch_y in batcher(X_train, y_train, batch_size):
            total_iter_num += 1
            model.train_on_batch(batch_X, batch_y)
            if total_iter_num % 100 == 0:
                train_score = model.evaluate(x=batch_X, y=batch_y, verbose=0)
                val_score = model.evaluate(x=X_test, y=y_test, verbose=0)
                print('train_loss is ', train_score, 'val_loss is ', val_score)

    val_score = model.evaluate(x=X_test, y=y_test, verbose=0)
    print('final test_loss is ', val_score)


if __name__=='__main__':
    housing = fetch_california_housing()
    X_data = housing.data
    y_data = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    print('X_train.shape is', X_train.shape, 'y_train.shape is', y_train.shape)
    print('X_test.shape is', X_test.shape, 'y_test.shape is', y_test.shape)
    regression_keras(X_train, y_train, X_test, y_test)

