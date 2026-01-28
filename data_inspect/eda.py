import pandas as pd

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
df.info()
df.describe()
df.isnull().sum()
df.value_counts(normalize=True)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nData Types:\n", df.dtypes)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop(columns=["customerID"])


df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTenure"] = (df["tenure"] > 24).astype(int)
categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
