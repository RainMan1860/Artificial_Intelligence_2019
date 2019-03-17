import os
import sys
import logging

import numpy as np
import pandas as pd

import tensorflow as tf

file_name = os.path.join('data', 'ex2data1.txt')
data = pd.read_table(file_name, sep=',', header=None, quoting=3)

# print(data.shape)

train_x = np.array(data.iloc[:, 0:2])
train_y = np.array(data.iloc[:, 2:3])

# print(train_x)
# print(train_y)

epoch = 1000
rate = 0.5

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

print(train_x)

# tensorflow model
x = tf.placeholder("float", [None, 2])
y = tf.placeholder("float", [None, 1])  

# w = tf.Variable(tf.random_normal([1])) # 生成随机权重，也就是我们的theta_1
w = tf.Variable(tf.random_uniform([2, 1], -1., 1.))
b = tf.Variable(tf.zeros([1])) # theta_0

y_pred = tf.nn.sigmoid(tf.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred, logits=y))

loss = tf.reduce_mean(- y * tf.log(y_pred) - (1 - y) * tf.log(1 - y_pred))
optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()  
sess.run(init)  
print('w start is ', sess.run(w).flatten())  
print('b start is ', sess.run(b).flatten())  
for index in range(epoch):  
    sess.run(optimizer, {x: train_x, y: train_y}) 

    if index % 10 == 0:
        print('w is', sess.run(w).flatten(), ' b is', sess.run(b).flatten(), ' loss is', sess.run(loss, {x: train_x, y: train_y}))

print('loss is ', sess.run(loss, {x: train_x, y: train_y})) 
print('w end is ',sess.run(w).flatten())  
print('b end is ',sess.run(b).flatten()) 
print('y_pred is ', sess.run(y_pred, {x: [[0.25, 0.25]]}))