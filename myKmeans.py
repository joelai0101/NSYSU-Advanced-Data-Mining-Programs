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

# 2. 實作 KMeans 演算法
def my_kmeans(X, n_clusters):
    # 隨機初始化群中心點
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    while True:
        # 計算每個點到各群中心點的距離
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        # 將每個點指派到最近的群中心點
        labels = np.argmin(distances, axis=0)
        # 重新計算群中心點
        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(n_clusters)])
        # 如果群中心點沒有變化，則結束迴圈
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

labels, centroids = my_kmeans(X, centers)

# print(labels)

# 繪製分群前的資料點散圖
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='blue', edgecolor='gray', label='Data Points')
plt.title('Before Clustering (My KMeans)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 繪製分群後的資料點散圖
plt.figure(figsize=(6, 6))
for i in range(centers):
    plt.scatter(X[labels==i, 0], X[labels==i, 1], s=50, c=colors[i], marker=markers[i], edgecolor='gray', label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='*', c='red', edgecolor='gray', label='Centroids')
plt.title('After Clustering (My KMeans)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()