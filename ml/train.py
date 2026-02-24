# file for orchestration only



from training import *


if __name__ == "__main__":
  df = load_data("../data/telco_customer_churn.csv")

  X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

  numerical, categorical = get_feature_types(X_train)

  preprocessor = build_preprocessor(numerical, categorical)

  pipeline = build_pipeline(preprocessor)

  pipeline.fit(X_train, y_train)

  save_model(pipeline)
