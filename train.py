import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.dropna()

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categoricals
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]

X = df[FEATURES]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("customer-churn-prediction")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")
