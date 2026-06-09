
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


    df = pd.read_csv("breast_cancer.csv")

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["diagnosis"] = df["diagnosis"].map({"M":1, "B":0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    st.success(f"Model Accuracy: {acc:.3f}")

    st.subheader("Predict New Sample")
    sample = []
    for col in X.columns:
        sample.append(st.number_input(col, value=float(X[col].mean())))

    if st.button("Predict"):
        pred = model.predict(scaler.transform([sample]))[0]
        st.write("Malignant" if pred == 1 else "Benign")
