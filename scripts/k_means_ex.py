# K_Means_Example.py

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic data with 4 clusters
X, y = make_blobs(n_samples=500, centers=4, random_state=42)

# Create a KMeans model with 4 clusters
model = KMeans(n_clusters=4)
model.fit(X)

# Get cluster assignments for each data point
labels = model.labels_

# Get the coordinates of cluster centers
cluster_centers = model.cluster_centers_

# Plot the clusters and their centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='k')
plt.title("K-Means Clustering")
plt.show()
