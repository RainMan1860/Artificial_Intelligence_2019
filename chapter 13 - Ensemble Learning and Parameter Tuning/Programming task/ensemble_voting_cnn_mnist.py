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
# y_train = np_utils.to_categorical(y_train)

batch_size = 128
num_classes = 10
epochs = 10

def create_model_cnn():

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
    return model

def create_model_mlp():
    input_layer = Input(shape=(784, ))
    hidden_layer_1 = Dense(256, activation='sigmoid') (input_layer)
    hidden_layer_2 = Dense(128, activation='sigmoid') (hidden_layer_1)
    output_layer = Dense(10, activation='softmax') (hidden_layer_2)

    model = Model(input_layer, output_layer)

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model

clf1 = KerasClassifier(build_fn=create_model_cnn, verbose=1, epochs=epochs, batch_size=batch_size)
clf2 = KerasClassifier(build_fn=create_model_mlp, verbose=1, epochs=epochs, batch_size=batch_size)

# from sklearn.ensemble import VotingClassifier
from vote_classifier import VotingClassifier
eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2)], voting='soft')
eclf1.fit(x_train, y_train)

y_pred = eclf1.predict(x_test)
print(y_pred)
save_file = os.path.join('result', 'ensemble_voting_cnn_keras.csv')
save_result(y_pred, file_name=save_file)