import os
import sys
import logging

import pandas as pd
import numpy as np

def load_train_data(file_name, normalize=True):
    x_train, y_train = [], []
    with open(file_name) as my_file:
        header = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_train.append(line[1:])
            y_train.append(int(line[0]))

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)

    if normalize == True:
        x_train /= 255

    return x_train, y_train

def load_test_data(file_name, normalize=True):
    x_test = []
    with open(file_name) as my_file:
        hader = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_test.append(line)

    x_test = np.array(x_test).astype('float32')
    if normalize == True:
        x_test /= 255

    return x_test

def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'ImageId': range(1, len(y_pred) + 1), 'Label': y_pred})
    result_df.to_csv(file_name, index=False)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_train(file_path, normalize=True):
    x_train = []
    y_train = []
    for i in range(5):
        file_name = os.path.join(file_path, 'data_batch_' + str(i+1))
        dict = unpickle(file_name)
        # print(dict.keys())
        for j in range(len(dict[b'data'])):
            x_train.append(dict[b'data'][j])
            y_train.append(dict[b'labels'][j])

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)
    if normalize == True:
        x_train /= 255

    return x_train, y_train

def load_cifar10_test(file_name, normalize=True):
    x_test = []
    y_test = []
    dict = unpickle(file_name)
    for j in range(len(dict[b'data'])):
        x_test.append(dict[b'data'][j])
        y_test.append(dict[b'labels'][j])

    x_test = np.array(x_test).astype('float32')
    y_test = np.array(y_test)
    if normalize == True:
        x_test /= 255

    return x_test, y_test
