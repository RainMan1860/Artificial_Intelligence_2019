import os
import sys
import logging
import pickle

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

batch_size = 128
num_classes = 10
epochs = 3
random_state = 0

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

print(x_train.shape)
print(x_dev.shape)
y_train = np_utils.to_categorical(y_train)
# y_dev = np_utils.to_categorical(y_dev)

input_layer = Input(shape=(784, ))
reshape_layer = Reshape((28, 28, 1)) (input_layer)
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (reshape_layer)
pooling_layer = MaxPooling2D(pool_size=(2, 2)) (conv_layer)
dropout_layer_1 = Dropout(0.25) (pooling_layer)
flatten_layer = Flatten()(dropout_layer_1)
hidden_layer = Dense(128, activation='relu') (flatten_layer)
dropout_layer_2 = Dropout(0.5) (hidden_layer)
output_layer = Dense(10, activation='softmax') (dropout_layer_2)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

y_pred = np.argmax(model.predict(x_dev), axis=-1)
print(y_pred.shape)
print(accuracy_score(y_dev, y_pred))
pickle_file = os.path.join('pickle', 'cnn_result.pickle')
pickle.dump(y_pred, open(pickle_file, 'wb'))