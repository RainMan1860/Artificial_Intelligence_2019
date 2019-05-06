import os
import sys
import logging

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
print('Total of records = ' + str(len(df)))

# construct user-item matix
data_matrix = np.zeros((n_users, n_items))
r_matrix = np.zeros((n_users, n_items))
for line in df.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
    r_matrix[line[1]-1, line[2]-1] = 1

print(data_matrix.shape)
print(r_matrix.shape) 

hidden_dim = 256

# construct user feature vector x
x = np.matrix(np.random.rand(n_users, hidden_dim))
theta = np.matrix(np.random.rand(n_items, hidden_dim))

print(x.shape)
print(theta.shape)

def cost_function(data_matrix, r_matrix, x, theta, reg_lambda=0.01):
    y_pred = x * theta.T

    error_sum = 0.
    for i in range(n_users):
        for j in range(n_items):
            if r_matrix[i, j] == 1.:
                error_sum = error_sum + (y_pred[i, j] - data_matrix[i, j]) ** 2

    error_sum = error_sum * 0.5

    x_regular = 0.
    for i in range(n_users):
        for k in range(hidden_dim):
            x_regular = x_regular + x[i, k] ** 2

    x_regular = reg_lambda * 0.5 * x_regular

    theta_regular = 0.
    for j in range(n_items):
        for k in range(hidden_dim):
            theta_regular = theta_regular + theta[j, k] ** 2

    theta_regular = reg_lambda * 0.5 * theta_regular

    return error_sum + x_regular + theta_regular

# print(cost_function(data_matrix, r_matrix, x, theta))

def gradient_descent_step(data_matrix, r_matrix, x, theta, reg_lambda=0.01, learning_rate=0.001):
    x_grad = np.zeros((n_users, hidden_dim))
    theta_grad = np.zeros((n_items, hidden_dim))

    error_matrix = x * theta.T - data_matrix
    # user grad
    for i in range(n_users):
        for k in range(hidden_dim):
            error = 0.

            for j in range(n_items):
                if r_matrix[i, j] == 1.:
                    error = error + error_matrix[i, j] * theta[j, k]

            x_grad[i, k] = error * theta[j, k] + reg_lambda * x[i, k]

    for j in range(n_items):
        for k in range(hidden_dim):
            error = 0.

            for i in range(n_users):
                if r_matrix[i, j] == 1.:
                    error = error + error_matrix[i, j] * x[i, k]

            theta_grad[j, k] = error * x [i, k] + reg_lambda * theta[j, k]

    x = x - learning_rate * x_grad
    theta = theta - learning_rate * theta_grad

    return x, theta


max_iter = 100
threshold = 0.001

pre_cost = cost_function(data_matrix, r_matrix, x, theta)
print('original cost: ' + str(pre_cost))
for i in range(max_iter):

    x, theta = gradient_descent_step(data_matrix, r_matrix, x, theta)
    cost = cost_function(data_matrix, r_matrix, x, theta)

    print("train step: %d, cost: %.2f" % (i, cost))
    if abs(cost - pre_cost) <= threshold:
        break

    pre_cost = cost

# pred
