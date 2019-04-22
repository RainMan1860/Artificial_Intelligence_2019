import os
import sys
import logging

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

kf = KFold(len(x_train), n_folds=5)
for train_index, dev_index in kf:
    print("TRAIN:", train_index, "DEV:", dev_index)
    X_train, Y_train = x_train[train_index], y_train[train_index]
    X_dev, Y_dev = x_train[dev_index], y_train[dev_index]
    clf = SVC()
    clf.fit(X_train, Y_train)
    Y_dev_pred = clf.predict(X_dev)

    print(accuracy_score(Y_dev, Y_dev_pred))
    print(f1_score(Y_dev, Y_dev_pred, average='macro'))