import os
import sys
import logging

import pickle
import numpy as np
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_train_data(file_path, normalize=True):
    x_train, y_train = [], []
    for i in range(5):
        file_name = os.path.join(file_path, 'data_batch_' + str(i+1))
        dict = unpickle(file_name)

        for j in range(len(dict[b'data'])):
            x_train.append(dict[b'data'][j])
            y_train.append(dict[b'labels'][j])

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)

    if normalize == True:
        x_train /= 255

    return x_train, y_train

def load_test_data(file_name, normalize=True):
    x_test, y_test = [], []

    dict = unpickle(file_name)
    for j in range(len(dict[b'data'])):
        x_test.append(dict[b'data'][j])
        y_test.append(dict[b'labels'][j])

    x_test = np.array(x_test).astype('float32')
    y_test = np.array(y_test)

    if normalize == True:
        x_test /= 255

    return x_test, y_test