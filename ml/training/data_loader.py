import pandas as pd

def load_data(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  df = df.drop_duplicates()

  # Convert target
  df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

  # Fix TotalCharges
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

  # Drop missing values
  df = df.dropna()

  # Drop  Useless Columns
  df = df.drop(columns=["customerID"])

  return df