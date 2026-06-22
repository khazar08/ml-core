import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

bandwidth = estimate_bandwidth(
    X_scaled,
    quantile=0.2,
    n_samples=len(X_scaled)
)


ms = MeanShift(bandwidth=bandwidth)

ms.fit(X_scaled)

labels = ms.labels_
cluster_centers = ms.cluster_centers_


# eval
n_clusters = len(np.unique(labels))

print("\nNumber of Clusters Found:", n_clusters)

if n_clusters > 1:
    score = silhouette_score(X_scaled, labels)
    print("Silhouette Score:", round(score, 3))


df['Cluster'] = labels

print(df['Cluster'].value_counts())


plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=labels,
    cmap='viridis',
    s=70
)

# Convert cluster centers back to original scale
centers_original = scaler.inverse_transform(cluster_centers)

plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    color='red',
    marker='X',
    s=300,
    label='Cluster Centers'
)

plt.title("Mean Shift Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.show()

# Results

# Estimated Bandwidth: 1.0467

# Number of Clusters Found: 3

# Silhouette Score: 0.4613

# Cluster Distribution:
# Cluster 0 : 121 customers
# Cluster 1 : 41 customers
# Cluster 2 : 38 customers

# Cluster Centers (Original Scale):
#
# Cluster 0
# Annual Income = 54.95 k$
# Spending Score = 48.74
#
# Cluster 1
# Annual Income = 80.23 k$
# Spending Score = 79.13
#
# Cluster 2
# Annual Income = 80.87 k$
# Spending Score = 21.21
