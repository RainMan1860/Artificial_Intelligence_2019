import os
import sys
import logging

import numpy as np

from util import load_train_data, load_test_data
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten
from keras.models import Model

from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

x_train, y_train = load_train_data('cifar-10')
x_test, y_test = load_test_data(os.path.join('cifar-10', 'test_batch'))

x_train = np.reshape(x_train, (x_train.shape[0], 32, 32, 3))
x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3))

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test_val = np_utils.to_categorical(y_test)

batch_size = 32
num_classes = 10
epochs = 200

x = Input(shape=(32, 32, 3))
y = x
y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

y = Flatten()(y)
y = Dropout(0.5)(y)
y = Dense(units=num_classes, activation='softmax', kernel_initializer='he_normal')(y)

model = Model(inputs=x, outputs=y, name='model1')

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

print(model.summary())

from datetime import datetime
from time import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def train(model, batch, epoch, data_augmentation=True):
    start = time()
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    
    es = EarlyStopping(monitor='val_acc', patience=20)
    mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5', 
                         monitor='val_acc', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    if data_augmentation:
        aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True)
        aug.fit(x_train)
        gen = aug.flow(x_train, y_train, batch_size=batch)
        h = model.fit_generator(generator=gen, 
                                 steps_per_epoch=50000/batch, 
                                 epochs=epoch, 
                                 validation_data=(x_test, y_test_val),
                                 callbacks=[es, mc, tb])
    else:
        start = time()
        h = model.fit(x=x_train, 
                      y=y_train, 
                      batch_size=batch, 
                      epochs=epoch, 
                      validation_data=(x_test, y_test),
                      callbacks=[es, mc, tb])
    
    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    return h

epoch = 200
batch_size = 64

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()

h = train(model, batch_size, epoch)
accuracy_curve(h)