import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def main():
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # NMF requires non-negative values.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    nmf = NMF(
        n_components=5,
        init="nndsvda",
        random_state=42,
        max_iter=1000
    )

    W = nmf.fit_transform(X_scaled)
    H = nmf.components_

    X_reconstructed = np.dot(W, H)

    mse = mean_squared_error(
        X_scaled.flatten(),
        X_reconstructed.flatten()
    )

    print("=" * 60)
    print("NMF RESULTS")
    print("=" * 60)

    print(f"Dataset samples   : {X.shape[0]}")
    print(f"Dataset features  : {X.shape[1]}")
    print(f"NMF components    : {nmf.n_components}")
    print(f"Reconstruction MSE: {mse:.6f}")

    print("\nW matrix shape:", W.shape)
    print("H matrix shape:", H.shape)

    print("\nTop features per component:")

    for i, component in enumerate(H):
        top_idx = np.argsort(component)[::-1][:5]

        print(f"\nComponent {i + 1}")

        for feature in X.columns[top_idx]:
            print(f"  - {feature}")

    pd.DataFrame(
        W,
        columns=[f"Component_{i + 1}" for i in range(nmf.n_components)]
    ).to_csv("nmf_latent_features.csv", index=False)

  
  # Reconstruction MSE: ~0.001-0.003
