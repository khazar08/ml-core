import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("/Users/khazar/Downloads/data 3.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

y = data["diagnosis"]
x = data.drop("diagnosis", axis=1)

# Split first so the test set stays unseen
# stratify=y keeps the malignant/benign ratio consistent across both sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# EVal
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")   # Accuracy: 97.08%
print(classification_report(y_test, y_pred)) 
