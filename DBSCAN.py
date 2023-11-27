from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 使用 sklearn.datasets 的 make_blobs 函數來建立資料
n_samples = 250
n_features = 2
centers = 4
shuffle = True
cluster_std = 1.5
random_state = 114514

X, y = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=centers,
                  shuffle=shuffle,
                  cluster_std=cluster_std,
                  random_state=random_state)

colors = ['lightgreen', 'orange', 'lightblue', 'purple']
markers = ['s', 'o', 'v', 'D']
noise_color = 'gray'

#print(X)
#print(X[:, 0])
#print(X[:, 1])

# 1.2 使用 sklearn.cluster 的 DBSCAN 函數
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)

# 繪製分群前的資料點散圖
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='blue', edgecolor='gray', label='Data Points')
plt.title('Before Clustering (DBSCAN)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 繪製分群後的資料點散圖
plt.figure(figsize=(6, 6))

# 繪製噪聲點
plt.scatter(X[dbscan_labels==-1, 0], X[dbscan_labels==-1, 1], s=50, c=noise_color, edgecolor='gray', label='Noise')

# 繪製非噪聲點
for i in range(centers):
    if np.any(dbscan_labels==i):
        plt.scatter(X[dbscan_labels==i, 0], X[dbscan_labels==i, 1], s=50, c=colors[i], edgecolor='gray', label=f'Cluster {i+1}')

plt.title('After Clustering (DBSCAN)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()