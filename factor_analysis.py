import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo
)


file = "data-final.csv"

question_cols = [c for c in df.columns if c.startswith("EXT") or
                                          c.startswith("EST") or
                                          c.startswith("AGR") or
                                          c.startswith("CSN") or
                                          c.startswith("OPN")]

X = df[question_cols].copy()



X.replace(0, np.nan, inplace=True)

X.dropna(inplace=True)

print("\nClean Shape:", X.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# bartlett test

chi_square_value, p_value = calculate_bartlett_sphericity(X)

print("\nBartlett Test")
print("Chi-Square =", round(chi_square_value, 2))
print("p-value =", p_value)

# kmo test

kmo_all, kmo_model = calculate_kmo(X)

print("\nKMO Score")
print(round(kmo_model, 3))

# initial factor analysis

fa = FactorAnalyzer(rotation=None)

fa.fit(X_scaled)

eigen_values, vectors = fa.get_eigenvalues()


plt.figure(figsize=(10, 5))

plt.scatter(
    range(1, len(eigen_values)+1),
    eigen_values
)

plt.plot(
    range(1, len(eigen_values)+1),
    eigen_values
)

plt.axhline(
    y=1,
    color="red",
    linestyle="--"
)

plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")

plt.show()

fa = FactorAnalyzer(
    n_factors=5,
    rotation="varimax"
)

fa.fit(X_scaled)

loadings = pd.DataFrame(
    fa.loadings_,
    index=question_cols,
    columns=[
        "Factor1",
        "Factor2",
        "Factor3",
        "Factor4",
        "Factor5"
    ]
)

print("\nFactor Loadings:")
print(loadings.round(3))

plt.figure(figsize=(12, 15))

sns.heatmap(
    loadings,
    annot=False,
    cmap="coolwarm",
    center=0
)

plt.title("Factor Loading Matrix")

plt.show()
communalities = pd.DataFrame(
    {
        "Question": question_cols,
        "Communality": fa.get_communalities()
    }
)

print("\nCommunalities")
print(communalities.head())

variance = pd.DataFrame(
    {
        "SS_Loadings": fa.get_factor_variance()[0],
        "Proportion": fa.get_factor_variance()[1],
        "Cumulative": fa.get_factor_variance()[2]
    }
)

print("\nVariance Explained")
print(variance)

scores = pd.DataFrame(
    fa.transform(X_scaled),
    columns=[
        "Factor1",
        "Factor2",
        "Factor3",
        "Factor4",
        "Factor5"
    ]
)

print("\nFactor Scores")
print(scores.head())

print("\nINTERPRETATION GUIDE")
print("- Factor with EXT questions -> Extraversion")
print("- Factor with EST questions -> Neuroticism")
print("- Factor with AGR questions -> Agreeableness")
print("- Factor with CSN questions -> Conscientiousness")
print("- Factor with OPN questions -> Openness")

# KMO = 0.913
# Bartlett p-value < 0.001
# Factors retained = 5
# Cumulative variance explained = 52.7%
# Interpretation: Factor 1 corresponds to Extraversion
