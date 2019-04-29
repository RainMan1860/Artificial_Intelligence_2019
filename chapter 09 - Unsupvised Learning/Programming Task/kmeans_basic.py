import scipy.io as sio
import numpy as np

import os
import sys
import logging

data_file = os.path.join('data', 'ex7data2.mat')
data = sio.loadmat(data_file)
X = data['X']
print(data['X'])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [4, 4]]))
print(kmeans.cluster_centers_)