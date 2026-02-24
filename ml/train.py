# file for orchestration only



from training import *


def main():
  df = load_data("ml/data/telco_customer_churn.csv")

  X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

  numerical, categorical = get_feature_types(X_train)

  preprocessor = build_preprocessor(numerical, categorical)

  pipeline = build_pipeline(preprocessor)
  # pipeline = build_random_forest_pipeline(preprocessor)

  #train the model
  model = pipeline.fit(X_train, y_train)

  #evaluate the model
  evaluate_model(model, X_val, y_val)
  
  #save the model
  save_model(model)

if __name__ == "__main__":
  main()
