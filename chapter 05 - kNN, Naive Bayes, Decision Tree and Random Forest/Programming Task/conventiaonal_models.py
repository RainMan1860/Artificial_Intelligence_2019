import os
import sys
import logging

import pandas as pd
import numpy as np

import tensorflow as tf
from util import load_train_data, load_test_data, save_result

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# save_file = os.path.join('result', 'k_neighbor.csv')

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100)
# save_file = os.path.join('result', 'random_forest.csv')

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# save_file = os.path.join('result', 'decision_tree.csv')

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
save_file = os.path.join('result', 'guassian_naive_bayes.csv')

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

save_file = os.path.join('result', 'guassian_naive_bayes.csv')
save_result(y_pred, save_file)