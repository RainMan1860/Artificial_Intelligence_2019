import os
import sys
import logging

import numpy as np
import pandas as pd

import tensorflow as tf

file_name = os.path.join('data', 'ex2data1.txt')
data = pd.read_table(file_name, sep=',', header=None, quoting=3)

# print(data.shape)

train_x = np.array(data.iloc[:, 0:2])
train_y = np.array(data.iloc[:, 2:3])

# print(train_x)
# print(train_y)

epoch = 1000
rate = 0.5

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x = np.insert(scaler.transform(train_x), 0, values=1., axis=1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_x, train_y)
print(clf.coef_)
print(clf.intercept_)

print(clf.predict([[1., 0.25, 0.25]]))
print(clf.predict_proba([[1., 0.25, 0.25]]))