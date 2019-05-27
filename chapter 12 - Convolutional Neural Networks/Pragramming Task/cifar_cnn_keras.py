import os
import sys
import logging

import numpy as np

from util import load_train_data, load_test_data
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

x_train, y_train = load_train_data('cifar-10')
x_test, y_test = load_test_data(os.path.join('cifar-10', 'test_batch'))

x_train = np.reshape(x_train, (x_train.shape[0], 32, 32, 3))
x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3))

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)

batch_size = 32
num_classes = 10
epochs = 100

input_layer = Input(shape=(32, 32, 3, ))
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (input_layer)
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
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))