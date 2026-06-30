import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main():
    wine = load_wine()

    X = pd.DataFrame(
        wine.data,
        columns=wine.feature_names
    )

    y = wine.target

    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    X_embedded = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y
    )

    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=wine.target_names,
        title="Wine Class"
    )

    plt.title("t-SNE Projection of Wine Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.tight_layout()
    plt.savefig("tsne_wine_visualization.png")
    plt.show()

    print(f"Samples           : {X.shape[0]}")
    print(f"Features          : {X.shape[1]}")
    print(f"Classes           : {len(wine.target_names)}")
    print(f"Embedding Shape   : {X_embedded.shape}")

    print("\nDataset Classes:")
    for idx, name in enumerate(wine.target_names):
        count = (y == idx).sum()
        print(f"  {name}: {count} samples")

    
  
"""

Results

Samples         : 178
Features        : 13
Classes         : 3
Embedding Shape : (178, 2)

Class Distribution:
- class_0: 59 samples
- class_1: 71 samples
- class_2: 48 samples

