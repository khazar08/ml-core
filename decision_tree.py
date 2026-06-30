import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv("garments_worker_productivity.csv")

df["wip"] = df["wip"].fillna(0)

for col in ["department", "quarter", "day"]:
    df[col] = df[col].str.strip()

df = df.drop(columns=["date"])

df_encoded = pd.get_dummies(df, columns=["department", "quarter", "day"], drop_first=True)

# Features (X) and target (y)
y = df_encoded["actual_productivity"]
X = df_encoded.drop(columns=["actual_productivity"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate on held-out data
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

# What the tree leaned on
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 5 features:")
print(importances.sort_values(ascending=False).head(5))
