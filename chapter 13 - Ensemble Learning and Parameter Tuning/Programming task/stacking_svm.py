import os
import sys
import logging
import pickle

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
# from keras.utils import np_utils
# from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
# from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 0

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100)

from sklearn.svm import SVC
clf = SVC()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_dev)
print(y_pred.shape)
print(accuracy_score(y_dev, y_pred))
pickle_file = os.path.join('pickle', 'svm_result.pickle')
pickle.dump(y_pred, open(pickle_file, 'wb'))