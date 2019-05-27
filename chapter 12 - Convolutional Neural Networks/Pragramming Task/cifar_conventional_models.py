import os
import sys
import logging

import numpy as np

from util import load_train_data, load_test_data
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

x_train, y_train = load_train_data('cifar-10')
x_test, y_test = load_test_data(os.path.join('cifar-10', 'test_batch'))

from sklearn.svm import SVC
clf = SVC()

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))

