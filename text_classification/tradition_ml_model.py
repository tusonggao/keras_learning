import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

def read_file_content(file_name):
    content = ''
    lines_lst = []
    with open(file_name) as file:
        for line in file:
            lines_lst.append(line.strip())
    return ' '.join(lines_lst)


def get_all_files(root_dir):
    lst = os.listdir(root_dir)
    file_names_lst = []
    for i in range(0, len(lst)):
        path = os.path.join(root_dir, lst[i])
        if os.path.isfile(path):
            # print('path is ', path)
            file_names_lst.append(lst[i])
    print('file_names_lst is', file_names_lst)
    return file_names_lst


def split_all_files(root_dir, train_nums, test_nums):
    file_names_lst = get_all_files(root_dir)

    train_files_lst, test_files_lst = [], []
    for name in file_names_lst:
        if int(name[2:5]) in train_nums:
            train_files_lst.append(root_dir + '/' + name)
        elif int(name[2:5]) in test_nums:
            test_files_lst.append(root_dir + '/' + name)

    print('len of train_files_lst', len(train_files_lst),
          'len of test_files_lst', len(test_files_lst))
    return train_files_lst, test_files_lst

def generate_data_from_raw_file():
    train_X, train_y, test_X, test_y = [], [], [], []

    nums_lst = np.arange(1000)
    np.random.seed(42)
    np.random.shuffle(nums_lst)
    train_nums, test_nums = nums_lst[:700], nums_lst[700:]
    print('train_nums[:10] is', train_nums[:10])

    train_files, test_files = split_all_files(
        'F:/using_keras/data/review_polarity/txt_sentoken/pos/',
        train_nums, test_nums)
    for file_name in train_files:
        train_X.append(read_file_content(file_name))
        train_y.append(1)
    for file_name in test_files:
        test_X.append(read_file_content(file_name))
        test_y.append(1)

    train_files, test_files = split_all_files(
        'F:/using_keras/data/review_polarity/txt_sentoken/neg/',
        train_nums, test_nums)
    for file_name in train_files:
        train_X.append(read_file_content(file_name))
        train_y.append(0)
    for file_name in test_files:
        test_X.append(read_file_content(file_name))
        test_y.append(0)

    return train_X, train_y, test_X, test_y


def data_preprocessing():
    print('in data_processing()')

    train_X, train_y, test_X, test_y = generate_data_from_raw_file()
    print('len of train_X is ', len(train_X))

    merged_X = train_X + test_X
    count_vect = CountVectorizer()
    merged_X = count_vect.fit_transform(merged_X)
    train_X, test_X = merged_X[:len(train_X)], merged_X[len(train_X):]
    train_y, test_y = np.array(train_y), np.array(test_y)

    # print('train_X.shape is', train_X.shape, 'test_X.shape is', test_X.shape)
    return train_X, train_y, test_X, test_y


def SGGClassifier_test(train_X, train_y, test_X, test_y):
    print('in SGGClassifier_test()')
    text_clf = SGDClassifier(loss='hinge', penalty='l2',alpha = 1e-3,
                             random_state = 42, max_iter = 5, tol = None)
    text_clf.fit(train_X, train_y)
    predicted_y = text_clf.predict(test_X)
    accuracy = np.mean(predicted_y == test_y)
    print('accuracy is ', accuracy)

def LinearSVC_test(train_X, train_y, test_X, test_y):
    print('in LinearSVC_test()')

    text_clf = SVC(kernel='linear')
    text_clf.fit(train_X, train_y)
    predicted_y = text_clf.predict(test_X)
    accuracy = np.mean(predicted_y == test_y)
    print('accuracy is ', accuracy)

def RBFSVC_test(train_X, train_y, test_X, test_y):
    print('in RBFSVC_test()')

    text_clf = SVC(kernel='rbf')
    text_clf.fit(train_X, train_y)
    predicted_y = text_clf.predict(test_X)
    accuracy = np.mean(predicted_y == test_y)
    print('accuracy is ', accuracy)


def MultinomialNB_test(train_X, train_y, test_X, test_y):
    print('in LinearSVC_test()')

    text_clf = MultinomialNB()
    text_clf.fit(train_X, train_y)
    predicted_y = text_clf.predict(test_X)
    accuracy = np.mean(predicted_y == test_y)
    print('accuracy is ', accuracy)


if __name__=='__main__':
    print('hello world!')
    # generate_data_from_raw_file()
    train_X, train_y, test_X, test_y = data_preprocessing()

    SGGClassifier_test(train_X, train_y, test_X, test_y)
    LinearSVC_test(train_X, train_y, test_X, test_y)
    RBFSVC_test(train_X, train_y, test_X, test_y)
    MultinomialNB_test(train_X, train_y, test_X, test_y)
