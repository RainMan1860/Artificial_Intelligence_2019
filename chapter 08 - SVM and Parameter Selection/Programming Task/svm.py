import os
import sys
import logging

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

from sklearn.svm import SVC
# C Penalty parameter C of the error term 
# kernel Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ (default=rbf)
clf = SVC()
save_file = os.path.join('result', 'svm.csv')

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
save_result(y_pred, save_file)