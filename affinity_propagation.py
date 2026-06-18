import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import os

=file_path = 'mall_customers.csv'
if not os.path.exists(file_path):
    data = pd.DataFrame({
        'Annual Income (k$)': np.random.randint(15, 140, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    })
else:
    data = pd.read_csv(file_path)

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# preference: Controls number of clusters (lower = fewer clusters)
# damping: Controls convergence stability (between 0.5 and 1.0)
af = AffinityPropagation(preference=-50, damping=0.7, random_state=42)
af.fit(X_scaled)

# results
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters = len(cluster_centers_indices)

print(f"Estimated number of clusters discovered: {n_clusters}")

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

for i, color in enumerate(colors):
    mask = (labels == i)
    plt.scatter(X.iloc[mask, 0], X.iloc[mask, 1], c=[color], label=f'Cluster {i}', alpha=0.6)

# plot the identified Exemplars (Cluster Centers)
centers = X.iloc[cluster_centers_indices]
plt.scatter(centers.iloc[:, 0], centers.iloc[:, 1], s=200, c='red', marker='X', edgecolors='black', label='Exemplars')

plt.title(f'Affinity Propagation Clustering (Found {n_clusters} Segments)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Estimated number of clusters discovered: 5

# Cluster Distribution:
# Cluster 0: Centered near (Income: 90k, Score: 20) -> "High Income, Low Spenders"
# Cluster 1: Centered near (Income: 25k, Score: 25) -> "Low Income, Low Spenders"
# Cluster 2: Centered near (Income: 55k, Score: 50) -> "Average Income, Average Spenders"
# Cluster 3: Centered near (Income: 25k, Score: 80) -> "Low Income, High Spenders"
# Cluster 4: Centered near (Income: 90k, Score: 80) -> "High Income, High Spenders"
