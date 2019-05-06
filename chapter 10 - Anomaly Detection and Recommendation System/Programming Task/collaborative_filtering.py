import os
import sys
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

nu = df.user_id.unique().shape[0]
nm = df.item_id.unique().shape[0]
print('Number of users = ' + str(nu) + ' | Number of movies = ' + str(nm))
print('Total of records = ' + str(len(df)))

Y = np.zeros((nu, nm))
R = np.zeros((nu, nm))
for line in df.itertuples():
    Y[line[1]-1, line[2]-1] = line[3]
    R[line[1]-1, line[2]-1] = 1

print('Average rating for movie 1 (Toy Story): %0.2f' % \
    np.mean([ Y[0][x] for x in range(Y.shape[1]) if R[0][x] ]))

# "Visualize the ratings matrix"
fig = plt.figure(figsize=(6,6*(1682./943.)))
dummy = plt.imshow(Y)
dummy = plt.colorbar()
dummy = plt.ylabel('Movies (%d)'% nm, fontsize=20)
dummy = plt.xlabel('Users (%d)'% nu, fontsize=20)
plt.show()