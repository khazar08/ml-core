import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

import umap.umap_ as umap

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
species_names = iris.target_names

print("Dataset Shape:", X.shape)
print("Species:", list(species_names))

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# umap

reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

embedding = reducer.fit_transform(X_scaled)

results = pd.DataFrame({
    "UMAP_1": embedding[:, 0],
    "UMAP_2": embedding[:, 1],
    "Species": [species_names[i] for i in y]
})

print(results.head())

plt.figure(figsize=(8, 6))

for species in species_names:
    subset = results[results["Species"] == species]

    plt.scatter(
        subset["UMAP_1"],
        subset["UMAP_2"],
        label=species,
        s=40
    )

plt.title("UMAP Projection of Iris Dataset")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()

plt.tight_layout()
plt.show()

# results

# UMAP successfully reduced the dimensionality of
# the Iris dataset from 4 features to 2 dimensions
# while maintaining species structure and revealing
# natural clustering patterns.
