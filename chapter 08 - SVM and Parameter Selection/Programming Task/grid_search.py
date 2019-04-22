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
# from sklearn.cross_validation import KFold
# from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.grid_search import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(x_train, y_train)

print(clf.grid_scores_)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)