import os
import sys
import logging

import pandas as pd
import numpy as np

from util import load_train_data, load_test_data, save_result
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

from keras.wrappers.scikit_learn import KerasClassifier

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

print(x_train.shape)

def create_model(activation='relu'):

    input_layer = Input(shape=(784, ))
    reshape_layer = Reshape((28, 28, 1)) (input_layer)
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation=activation) (reshape_layer)
    pooling_layer = MaxPooling2D(pool_size=(2, 2)) (conv_layer)
    dropout_layer_1 = Dropout(0.25) (pooling_layer)
    flatten_layer = Flatten()(dropout_layer_1)
    hidden_layer = Dense(128, activation=activation) (flatten_layer)
    dropout_layer_2 = Dropout(0.5) (hidden_layer)
    output_layer = Dense(10, activation='softmax') (dropout_layer_2)

    model = Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

batch_size = 128
num_classes = 10
epochs = 10

model = KerasClassifier(build_fn=create_model, nb_epoch=epochs, batch_size=batch_size, verbose=1)

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)

from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
