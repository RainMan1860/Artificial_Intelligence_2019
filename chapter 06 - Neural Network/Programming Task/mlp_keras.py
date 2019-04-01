import os
import sys
import logging

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
from keras.utils import np_utils
from keras.layers import Dense, Input
from keras.models import Model


train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

batch_size = 100
nb_epoch = 20
hidden_units_1 = 256
hidden_units_2 = 100

y_train = np_utils.to_categorical(y_train)

input_layer = Input(shape=(784, ))
hidden_layer_1 = Dense(hidden_units_1, activation='sigmoid') (input_layer)
hidden_layer_2 = Dense(hidden_units_2, activation='sigmoid') (hidden_layer_1)
output_layer = Dense(10, activation='softmax') (hidden_layer_2)

model = Model(input_layer, output_layer)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)

y_pred = np.argmax(model.predict(x_test), axis=-1)
print(y_pred)
save_file = os.path.join('result', 'mlp_keras.csv')
save_result(y_pred, file_name=save_file)



