import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA


def main():
    wine = load_wine()

    X = wine.data
    y_true = wine.target
    feature_names = wine.feature_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    birch = Birch(
        threshold=0.7,
        branching_factor=50,
        n_clusters=3,
    )

    labels = birch.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels

    cluster_sizes = df["Cluster"].value_counts().sort_index()
    cluster_means = df.groupby("Cluster")[feature_names].mean()
    

    print(f"\nSamples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Target Classes: {len(np.unique(y_true))}")

    print("\nCluster Sizes")
    print(cluster_sizes)

    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Adjusted Rand Index    : {ari:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")

    print(cluster_means.round(2))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "PC1": reduced[:, 0],
            "PC2": reduced[:, 1],
            "Cluster": labels,
        }
    )

    print("\nFirst 10 PCA-Reduced Samples")
    print(pca_df.head(10).round(3))

    print("\nNumber of CF Subclusters Created:")
    print(len(birch.subcluster_centers_))

    print("\nFirst 5 Subcluster Centers")
    print(np.round(birch.subcluster_centers_[:5], 3))


if __name__ == "__main__":
    main()


# REsults

# BIRCH Parameters:
# - threshold = 0.7
# - branching_factor = 50
# - n_clusters = 3
#
# Actual Results:
# - Cluster Sizes: [56, 65, 57]
# - Silhouette Score: 0.2713
# - Adjusted Rand Index: 0.7137
# - Calinski-Harabasz Score: 68.2318
# - CF Subclusters Created: 163
