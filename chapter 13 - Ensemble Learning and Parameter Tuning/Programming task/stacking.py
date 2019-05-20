import os
import sys
import logging
import pickle

import numpy as np
import pandas as pd

from util import load_train_data, load_test_data, save_result
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

stacking_result_files = [os.path.join('pickle', 'cnn_result.pickle'), 
                   os.path.join('pickle', 'rf_result.pickle'),
                   os.path.join('pickle', 'svm_result.pickle')]

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

random_state = 0

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

results = []
for i in range(len(stacking_result_files)):
    res = pickle.load(open(stacking_result_files[i], 'rb'))

    results.append(res)

x_stacking = []
for i in range(len(stacking_result_files)):
    x_sta = []
    for j in range(len(y_dev)):
        x_ij = results[i][j]
        x_sta.append(x_ij)

    x_stacking.append(x_sta)

x_stacking = np.array(x_stacking, dtype='float32')
print(x_stacking.shape)
print(y_dev.shape)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_dev, y_dev)
y_devpred = model.predict(x_dev)
print(y_devpred)
print(accuracy_score(y_dev, y_devpred))

y_pred = model.predict(x_test)
save_file = os.path.join('result', 'stacking_cnn_svm_rf.csv')
save_result(y_pred, save_file)