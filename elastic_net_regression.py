
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


df = pd.read_csv("Housing.csv")

target = "price"
X = df.drop(columns=[target])
y = df[target]

numeric = X.select_dtypes(include=["int64","float64"]).columns
categorical = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = ElasticNet(max_iter=10000)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

params = {
    "model__alpha":[0.01,0.1,1,10],
    "model__l1_ratio":[0.2,0.5,0.8]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid = GridSearchCV(pipe, params, cv=5, scoring="r2")
grid.fit(X_train, y_train)

pred = grid.predict(X_test)

st.write("Best Alpha:", grid.best_params_["model__alpha"])
st.write("Best L1 Ratio:", grid.best_params_["model__l1_ratio"])
st.write("R² Score:", round(r2_score(y_test, pred), 4))
st.write("MAE:", round(mean_absolute_error(y_test, pred), 2))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))


# Best Alpha: 0.1
# Best L1 Ratio: 	0.5
# R² Score:	0.6434
# MAE: 981,080.70
# RMSE: 1,342,464.18

