import os
import sys
import logging

import numpy as np
import pandas as pd

from keras.layers import Dense, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

file_name = os.path.join('data', 'ex2data1.txt')
data = pd.read_table(file_name, sep=',', header=None, quoting=3)

epoch = 100
batch_size = 2
# print(data.shape)

train_x = np.array(data.iloc[:, 0:2])
train_y = to_categorical(data.iloc[:, 2:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

x = Input(shape=(2, ))
y = Dense(2, activation='softmax')(x)

model = Model(x, y)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,)

print(model.layers[1].get_weights())
print(model.predict(np.array([[0.25, 0.25]])))