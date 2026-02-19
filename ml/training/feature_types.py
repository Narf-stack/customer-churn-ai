def get_feature_types(X):
  numerical_features = X.select_dtypes(
    include=["int64", "float64"]
  ).columns.tolist()

  categorical_features = X.select_dtypes(
    include=["object", "string"]
  ).columns.tolist()

  return numerical_features, categorical_features
