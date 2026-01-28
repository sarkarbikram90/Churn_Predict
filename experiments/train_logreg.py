import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------------------
# Load & clean data
# ----------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop(columns=["customerID"])

# Encode categorical
df["Contract"] = LabelEncoder().fit_transform(df["Contract"])

# Feature engineering
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTenure"] = (df["tenure"] > 24).astype(int)

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "AvgMonthlySpend",
    "IsLongTenure"
]

X = df[FEATURES]
y = df["Churn"].map({"Yes": 1, "No": 0})

# Scale for Logistic Regression
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MLflow experiment
# ----------------------------
mlflow.set_experiment("churn-model-comparison")

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    mlflow.log_param("model_type", "logistic_regression")

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

    mlflow.sklearn.log_model(model, "model")
