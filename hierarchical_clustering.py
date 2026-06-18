import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")

print(df.isnull().sum())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])


plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)

plt.title("Customer Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.grid(True)
plt.show()

#Scatter plot shows several visible customer groups, so clustering can be applied successfully.

plt.figure(figsize=(12, 6))

linked = linkage(
    X_scaled,
    method="ward"
)

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")

plt.show()


model = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

print(df["Cluster"].value_counts().sort_index())



plt.figure(figsize=(10, 7))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.title("Hierarchical Clustering Results")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()

"""

Five customer groups become visible.

Cluster characteristics:

Cluster 0
-----------
High Income
High Spending

Cluster 1
-----------
High Income
Low Spending

Cluster 2
-----------
Medium Income
Medium Spending

Cluster 3
-----------
Low Income
High Spending

Cluster 4
-----------
Low Income
Low Spending
"""


# cluster analysis
cluster_summary = df.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()


df.to_csv("clustered_customers.csv", index=False)

print("clustered_customers.csv")
