import os
import sys
import logging

import numpy as np

from util import load_train_data, load_test_data
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

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
# reshape_layer = Reshape((32, 32, 3)) (input_layer)
conv_layer_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (input_layer)
conv_layer_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (conv_layer_1)
pooling_layer_1 = MaxPooling2D(pool_size=(2, 2)) (conv_layer_2)
dropout_layer_1 = Dropout(0.25) (pooling_layer_1)

conv_layer_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu') (dropout_layer_1)
conv_layer_4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu') (conv_layer_3)
pooling_layer_2 = MaxPooling2D(pool_size=(2, 2)) (conv_layer_4)
dropout_layer_2 = Dropout(0.25) (pooling_layer_2)

flatten_layer = Flatten()(dropout_layer_2)
hidden_layer = Dense(512, activation='relu') (flatten_layer)
dropout_layer_2 = Dropout(0.5) (hidden_layer)
output_layer = Dense(num_classes, activation='softmax') (dropout_layer_2)

model = Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Using real-time data augmentation.')

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        epochs=epochs,
                        workers=4)

y_pred = np.argmax(model.predict(x_test), axis=-1)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
