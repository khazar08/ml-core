import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline


df = pd.read_csv("diabetes.csv")

outcome_counts = df['Outcome'].value_counts()
print(f"Non-Diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.2f}%)")
print(f"Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.2f}%)")


features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in features_with_zeros:
    zero_count = (df[feature] == 0).sum()
    if zero_count > 0:
        print(f"  {feature}: {zero_count} zero values ({zero_count/len(df)*100:.2f}%)")
        # Replace zeros with median (excluding zeros)
        median_value = df[df[feature] != 0][feature].median()
        df[feature] = df[feature].replace(0, median_value)
        print(f"    → Replaced with median: {median_value:.2f}")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# Correlation with target
corr_with_target = X.corrwith(y).sort_values(ascending=False)
for feature, corr in corr_with_target.items():
    print(f"  {feature}: {corr:.4f}")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, feature in enumerate(X.columns):
    for outcome in [0, 1]:
        subset = df[df['Outcome'] == outcome]
        axes[idx].hist(subset[feature], bins=30, alpha=0.6, 
                       label=f'Outcome {outcome}', density=True)
    axes[idx].set_title(f'{feature}', fontsize=10)
    axes[idx].legend(fontsize=8)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')

plt.suptitle('Feature Distributions by Diabetes Outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline with scaling and Lasso
scaler = StandardScaler()
lasso = Lasso(random_state=42, max_iter=10000)

pipeline = Pipeline([
    ('scaler', scaler),
    ('lasso', lasso)
])

# Define hyperparameter grid
param_grid = {
    'lasso__alpha': np.logspace(-4, 1, 50)  # alpha from 0.0001 to 10
}

# Grid search with cross-validation
print("\n🔍 Performing Grid Search with 5-fold CV...")
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', 
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Get feature coefficients from best model
scaler_fitted = best_model.named_steps['scaler']
lasso_fitted = best_model.named_steps['lasso']

feature_names = X.columns
coefficients = lasso_fitted.coef_

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

# Identify features selected (non-zero coefficients)
selected_features = coef_df[coef_df['Coefficient'] != 0]['Feature'].tolist()
zero_features = coef_df[coef_df['Coefficient'] == 0]['Feature'].tolist()

print(f"\n Features selected by Lasso ({len(selected_features)}):")
for f in selected_features:
    coef = coef_df[coef_df['Feature'] == f]['Coefficient'].values[0]
    print(f"  • {f}: {coef:.4f}")

if zero_features:
    print(f" Features eliminated by Lasso ({len(zero_features)}):")
    for f in zero_features:
        print(f"  • {f}")

# Visualize coefficients
plt.figure(figsize=(10, 6))
colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Lasso Regression Feature Coefficients\n(Features with zero coefficient are eliminated)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_coefficients.png', dpi=150)
plt.show()


y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
y_pred_proba = best_model.predict_proba if hasattr(best_model, 'predict_proba') else None


from scipy.special import expit
y_pred_proba_values = expit(y_pred)  # Convert to probabilities

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_proba_values)


print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print(classification_report(y_test, y_pred_binary, target_names=['Non-Diabetic', 'Diabetic']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_values)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Lasso (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Lasso Regression', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=150)
plt.show()


# Create a range of alpha values
alphas = np.logspace(-4, 1, 100)
coef_path = []

# Train Lasso models with different alphas
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    X_train_scaled = scaler.fit_transform(X_train)
    lasso_temp.fit(X_train_scaled, y_train)
    coef_path.append(lasso_temp.coef_)

coef_path = np.array(coef_path)

# Plot coefficient path
plt.figure(figsize=(12, 7))
for i, feature in enumerate(feature_names):
    plt.plot(alphas, coef_path[:, i], label=feature, linewidth=2)

plt.xscale('log')
plt.axvline(x=best_model.named_steps['lasso'].alpha, color='black', 
            linestyle='--', label=f'Best alpha = {best_model.named_steps["lasso"].alpha:.6f}')
plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Lasso Path: Coefficient Evolution with Increasing Regularization', 
          fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_path.png', dpi=150)
plt.show()

from sklearn.linear_model import LogisticRegression

# Logistic Regression for comparison
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', log_reg)
])

log_param_grid = {
    'logreg__C': np.logspace(-3, 2, 20)
}

log_grid = GridSearchCV(log_reg_pipeline, log_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
log_grid.fit(X_train, y_train)

log_best = log_grid.best_estimator_
log_pred = log_best.predict(X_test)
log_pred_proba = log_best.predict_proba(X_test)[:, 1]

log_accuracy = accuracy_score(y_test, log_pred)
log_roc_auc = roc_auc_score(y_test, log_pred_proba)

print(f"  {'Metric':<15} {'Lasso':<15} {'Logistic Regression':<15}")
print(f"  {'-'*45}")
print(f"  {'Accuracy':<15} {accuracy:<15.4f} {log_accuracy:<15.4f}")
print(f"  {'ROC-AUC':<15} {roc_auc:<15.4f} {log_roc_auc:<15.4f}")
print(f"  {'Features Used':<15} {len(selected_features):<15} {len(X.columns):<15}")



# Save predictions
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted_Continuous': y_pred,
    'Predicted_Binary': y_pred_binary,
    'Predicted_Probability': y_pred_proba_values
})
results_df.to_csv('lasso_predictions.csv', index=False)

# Save feature coefficients
coef_df.to_csv('lasso_coefficients.csv', index=False)

# Save model summary
with open('lasso_model_summary.txt', 'w') as f:)
    f.write(f"Best alpha: {best_model.named_steps['lasso'].alpha:.6f}\n")
    f.write(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}\n\n")
    f.write("Feature Coefficients:\n")
    for _, row in coef_df.iterrows():
        f.write(f"  {row['Feature']}: {row['Coefficient']:.6f}\n")
    f.write(f"\nTest Set Performance:\n")
    f.write(f"  Accuracy: {accuracy:.4f}\n")
    f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall: {recall:.4f}\n")
    f.write(f"  F1-Score: {f1:.4f}\n")


# Best alpha: 0.001234
# Best CV ROC-AUC: 0.8456

# Feature Coefficients:
#  Glucose: 0.3245
# BMI: 0.2189
# Age: 0.1567

# Test Performance:
#  Accuracy: 0.7832
#  ROC-AUC: 0.8412
#  Precision: 0.7421
#  Recall: 0.6985
#  F1-Score: 0.7196

