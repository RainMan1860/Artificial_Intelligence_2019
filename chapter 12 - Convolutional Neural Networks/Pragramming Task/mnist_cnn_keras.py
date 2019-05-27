import os
import sys
import logging

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

# x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
print(x_train.shape)
y_train = np_utils.to_categorical(y_train)

batch_size = 100
num_classes = 10
epochs = 12

input_layer = Input(shape=(784, ))
reshape_layer = Reshape((28, 28, 1)) (input_layer)
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (reshape_layer)
pooling_layer = MaxPooling2D(pool_size=(2, 2)) (conv_layer)
dropout_layer_1 = Dropout(0.25) (pooling_layer)
flatten_layer = Flatten()(dropout_layer_1)
hidden_layer = Dense(128, activation='relu') (flatten_layer)
dropout_layer_2 = Dropout(0.5) (hidden_layer)
output_layer = Dense(num_classes, activation='softmax') (dropout_layer_2)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

y_pred = np.argmax(model.predict(x_test), axis=-1)
save_file = os.path.join('result', 'cnn_keras.csv')
save_result(y_pred, file_name=save_file)