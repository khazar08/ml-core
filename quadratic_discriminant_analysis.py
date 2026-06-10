import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay)
import seaborn as sns



data = pd.read_csv("wine.csv")
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# QDA model
qda = QuadraticDiscriminantAnalysis(
    reg_param=0.0,      # Regularization 
    store_covariance=True  # Store covariance matrices per class
)

qda.fit(X_train_scaled, y_train)

# Eval
y_pred = qda.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - QDA on Wine Dataset')
plt.show()

# Cross-validation
cv_scores = cross_val_score(qda, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores (5-fold):")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Individual folds: {cv_scores}")

# 8. Param. inspection

print(f"Number of classes: {qda.classes_}")
print(f"Class priors: {np.round(qda.priors_, 3)}")
print(f"Number of features: {qda.means_.shape[1]}")

print(f"\nCovariance matrices per class (shape):")
for i, class_name in enumerate(target_names):
    cov_shape = qda.covariance_[i].shape
    print(f"  {class_name}: {cov_shape}")

X_2d = X_train_scaled[:, :2] 
X_test_2d = X_test_scaled[:, :2]

qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_2d, y_train)

# Meshgrid for decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = qda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, 
                          cmap='Set1', edgecolor='k', s=50)
axes[0].set_xlabel(f'{feature_names[0]} (scaled)')
axes[0].set_ylabel(f'{feature_names[1]} (scaled)')
axes[0].set_title('QDA Decision Boundaries (Training Data)')
axes[0].legend(*scatter.legend_elements(), title="Wine Classes")

axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, 
                cmap='Set1', edgecolor='k', s=80, alpha=0.7)
axes[1].set_xlabel(f'{feature_names[0]} (scaled)')
axes[1].set_ylabel(f'{feature_names[1]} (scaled)')
plt.tight_layout()
plt.show()

# LDA comparison
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)

print(f"QDA Accuracy: {accuracy:.4f}")
print(f"LDA Accuracy: {acc_lda:.4f}")
print(f"Difference: {(accuracy - acc_lda):+.4f}")

# 11. regularization tuning
reg_params = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
cv_scores_reg = []

for reg in reg_params:
    qda_reg = QuadraticDiscriminantAnalysis(reg_param=reg)
    scores = cross_val_score(qda_reg, X_train_scaled, y_train, cv=5)
    cv_scores_reg.append(scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(reg_params, cv_scores_reg, 'o-', linewidth=2, markersize=8)
plt.xlabel('Regularization Parameter (reg_param)')
plt.ylabel('Cross-validation Accuracy')
plt.title('Effect of Regularization on QDA Performance')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nBest regularization parameter: {reg_params[np.argmax(cv_scores_reg)]}")
print(f"Best CV accuracy: {max(cv_scores_reg):.4f}")

# Dataset shape: (178, 13)
# Number of classes: 3 - ['class_0' 'class_1' 'class_2']
# Features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
#           'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
#           'proanthocyanins', 'color_intensity', 'hue', 
#           'od280/od315_of_diluted_wines', 'proline']

# Training samples: 124
# Test samples: 54


# Test Accuracy: 0.9815

# Classification Report:
#              precision    recall  f1-score   support

#     class_0       1.00      0.94      0.97        18
#     class_1       0.95      1.00      0.97        19
#     class_2       1.00      1.00      1.00        17

#   accuracy                           0.98        54
#  macro avg       0.98      0.98      0.98        54
# weighted avg       0.98      0.98      0.98        54

# Cross-validation scores (5-fold):
#  Mean: 0.9768 (+/- 0.0234)
# Individual folds: [0.96 1.00 0.96 0.96 1.00]


# Covariance matrices per class (shape):
#  class_0: (13, 13)
#  class_1: (13, 13)
#  class_2: (13, 13)

# QDA vs LDA
# QDA Accuracy: 0.9815
# LDA Accuracy: 0.9630
# Difference: +0.0185

# Best regularization parameter: 0.01
# Best CV accuracy: 0.9848
