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
    # Real dataset with 178 samples, 13 chemical features, and 3 classes
    wine = load_wine()

    X = wine.data
    y_true = wine.target
    feature_names = wine.feature_names

    # Feature scaling is important for distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # BIRCH clustering
    birch = Birch(
        threshold=0.7,
        branching_factor=50,
        n_clusters=3,
    )

    labels = birch.fit_predict(X_scaled)

    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    # Cluster summary
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels

    cluster_sizes = df["Cluster"].value_counts().sort_index()
    cluster_means = df.groupby("Cluster")[feature_names].mean()

    print("=" * 60)
    print("BIRCH CLUSTERING ON WINE DATASET")
    print("=" * 60)

    print(f"\nSamples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Target Classes: {len(np.unique(y_true))}")

    print("\nCluster Sizes")
    print(cluster_sizes)

    print("\nEvaluation Metrics")
    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Adjusted Rand Index    : {ari:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")

    print("\nCluster Feature Means")
    print(cluster_means.round(2))

    # Simple 2D representation for inspection
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
#
# Observations:
# - ARI of 0.7137 indicates good agreement with the true wine classes.
# - BIRCH successfully identified 3 major groups in the dataset.
# - The algorithm compressed the dataset into 163 CF subclusters
#   before performing final clustering.
# - Cluster 1 contains wines with the highest average Proline level
#   (~1070), while Cluster 2 contains the lowest (~498).
# - Feature scaling was essential for obtaining meaningful clusters.
