from sklearn.model_selection import train_test_split

def split_data(df):
  X = df.drop("Churn", axis=1)
  y = df["Churn"]

  X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
  )

  X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
  )

  return X_train, X_val, X_test, y_train, y_val, y_test
