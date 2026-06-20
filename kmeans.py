import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
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

    kmeans = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    labels = kmeans.fit_predict(X_scaled)

    # eval
    silhouette = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    # Cluster summary
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels

    cluster_sizes = df["Cluster"].value_counts().sort_index()
    cluster_means = df.groupby("Cluster")[feature_names].mean()

    print(f"\nSamples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Target Classes: {len(np.unique(y_true))}")

    print("\nCluster Sizes")
    print(cluster_sizes)

    print("\nEvaluation Metrics")
    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Adjusted Rand Index    : {ari:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")

    print("\nCluster Feature Means")
    print(cluster_means.round(2))

    # PCA projection
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "PC1": reduced[:, 0],
            "PC2": reduced[:, 1],
            "Cluster": labels,
        }
    )

    print(pca_df.head(10).round(3))

    print("\nCluster Centers (Scaled Feature Space)")
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=feature_names,
    )
    print(centers.round(3))

    print("\nInertia (Within-Cluster Sum of Squares)")
    print(round(kmeans.inertia_, 4))


if __name__ == "__main__":
    main()


# Results
# - Cluster Sizes: [62, 51, 65]
# - Silhouette Score: 0.2849
# - Adjusted Rand Index: 0.8975
# - Calinski-Harabasz Score: 70.9400
# - Inertia: 1277.93
# - K-Means achieved an ARI of 0.8975, indicating very strong agreement
#   with the true wine classes.
# - The Silhouette Score (0.2849) is slightly better than the BIRCH run.
# - K-Means performed better than BIRCH on this dataset in terms of
#   cluster purity and alignment with the actual wine categories.
