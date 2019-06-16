import os
import sys
import logging

import numpy as np
from util import load_train_data

import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Activation, Conv2D, BatchNormalization, Dropout, Reshape, Flatten, UpSampling2D, ZeroPadding2D

def create_generator(latent_dim):
    sequence = Input(shape=(latent_dim, ), dtype='float32')
    model = Dense(128 * 7 * 7, activation='relu')(sequence)
    model = Reshape((7, 7, 128))(model)
    model = UpSampling2D()(model)
    model = Conv2D(128, kernel_size=3, padding='same')(model)
    model = BatchNormalization(momentum=0.8) (model)
    model = Activation('relu')(model)
    model = UpSampling2D()(model)
    model = Conv2D(64, kernel_size=3, padding='same')(model)
    model = BatchNormalization(momentum=0.8) (model)
    model = Activation('relu')(model) 
    model = Conv2D(channels, kernel_size=3, padding='same')(model)
    output = Activation('sigmoid')(model)

    model = Model(inputs=sequence, outputs=output)
    model.summary()

    return model

def create_discriminator(img_shape=(28, 28, 1)):
    sequence = Input(shape=img_shape)
    # model = Flatten()(sequence)
    # model = Dense(512)(model)
    # model = LeakyReLU(alpha=0.2)(model)
    # model = Dense(256)(model)
    # model = LeakyReLU(alpha=0.2)(model)
    # output = Dense(1, activation='sigmoid')(model)

    model = Conv2D(32, kernel_size=3, strides=2, padding='same')(sequence)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25) (model)
    model = Conv2D(64, kernel_size=3, strides=2, padding='same')(model)
    model = ZeroPadding2D(padding=((0, 1), (0, 1))) (model)
    model = BatchNormalization(momentum=0.8) (model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25) (model)
    model = Conv2D(128, kernel_size=3, strides=2, padding='same')(model)
    model = BatchNormalization(momentum=0.8) (model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25) (model)
    model = Conv2D(256, kernel_size=3, strides=2, padding='same')(model)
    model = BatchNormalization(momentum=0.8) (model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.25) (model)
    model = Flatten() (model)
    output = Dense(1, activation='sigmoid') (model)

    model = Model(inputs=sequence, outputs=output)
    model.summary()

    return model

def sample_images(epoch, latent_dim, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    print(gen_imgs.shape)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()


train_file = os.path.join('data', 'train.csv')
x_train, y_train = load_train_data(train_file)
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

batch_size = 32
epochs = 4000
latent_dim = 28*28
sample_interval = 50
channels = 1

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

discriminator.trainable = False

generator = create_generator(latent_dim, )
z = Input(shape=(latent_dim, ))
img = generator(z)

validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')


for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    g_loss = combined.train_on_batch(noise, valid)

    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    if epoch % sample_interval == 0:
        sample_images(epoch, latent_dim, generator)