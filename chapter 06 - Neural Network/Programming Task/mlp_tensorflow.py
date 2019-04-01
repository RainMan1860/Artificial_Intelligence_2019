import os
import sys
import logging

import tensorflow as tf
import numpy as np

from util import load_train_data, load_test_data, save_result
from keras.utils import np_utils

batch_size = 100
nb_epoch = 10

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
y_train = np_utils.to_categorical(y_train)
x_test = load_test_data(test_file)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

h1_W = tf.Variable(tf.random_normal([784, 256]))
h1_b = tf.Variable(tf.random_normal([256]))

h2_W = tf.Variable(tf.random_normal([256, 100]))
h2_b = tf.Variable(tf.random_normal([100]))

out_W = tf.Variable(tf.random_normal([100, 10]))
out_b = tf.Variable(tf.random_normal([10]))

hidden_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h1_W), h1_b))
hidden_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, h2_W), h2_b))
y_ = tf.matmul(hidden_layer_2, out_W) + out_b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_epoch):
        avg_cost = 0.
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            batch_xs = x_train[i*batch_size:(i+1)*batch_size]
            batch_ys = y_train[i*batch_size:(i+1)*batch_size]

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

        print('epoch: %d, cost: %.9f' % (epoch+1, avg_cost))

    y_pred = sess.run(y_, {x: x_test})

y_pred = np.argmax(y_pred, axis=-1)
save_file = os.path.join('result', 'mlp_tensordlow.csv')
save_result(y_pred, save_file)