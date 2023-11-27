# 引入必要的函式庫
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

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

# print(X)
# print(X[:, 0])
# print(X[:, 1])

# 3. 使用 sklearn.cluster 的 KMeans 函數
kmeans = KMeans(n_clusters=centers,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# print(kmeans_labels)

# 繪製分群前的資料點散圖
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='blue', edgecolor='gray', label='Data Points')
plt.title('Before Clustering (sklearn KMeans)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 繪製分群後的資料點散圖
plt.figure(figsize=(6, 6))
for i in range(centers):
    plt.scatter(X[kmeans_labels==i, 0], X[kmeans_labels==i, 1], s=50, c=colors[i], marker=markers[i], edgecolor='gray', label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='gray', label='Centroids')
plt.title('After Clustering (sklearn KMeans)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()