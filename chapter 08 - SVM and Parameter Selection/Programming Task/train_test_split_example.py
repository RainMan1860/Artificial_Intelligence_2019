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

print('Before split...')
print(x_train.shape)
print(y_train.shape)

X_train, X_dev, y_train, y_dev = train_test_split(
    x_train, y_train, test_size=0.2, random_state=0)

print('After split...')
print(X_train.shape)
print(X_dev.shape)
print(y_train.shape)
print(y_dev.shape)

from sklearn.svm import SVC
# C Penalty parameter C of the error term 
# kernel Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ (default=rbf)
clf = SVC()
clf.fit(X_train, y_train)
y_dev_pred = clf.predict(X_dev)

print(precision_score(y_dev, y_dev_pred, average='macro'))
print(precision_score(y_dev, y_dev_pred, average='micro'))
print(precision_score(y_dev, y_dev_pred, average=None))

print(recall_score(y_dev, y_dev_pred, average='macro'))
print(recall_score(y_dev, y_dev_pred, average='micro'))
print(recall_score(y_dev, y_dev_pred, average=None))

print(accuracy_score(y_dev, y_dev_pred))

print(f1_score(y_dev, y_dev_pred, average='macro'))
print(f1_score(y_dev, y_dev_pred, average='micro'))
print(f1_score(y_dev, y_dev_pred, average=None))