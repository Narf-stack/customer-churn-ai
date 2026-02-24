# file for orchestration only



from training import *


def main():
  df = load_data("ml/data/telco_customer_churn.csv")

  X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

  numerical, categorical = get_feature_types(X_train)

  preprocessor = build_preprocessor(numerical, categorical)

  logistic_model = build_pipeline(preprocessor)
  rf_model = build_random_forest_pipeline(preprocessor)
  gb_model = build_gradient_boosting_pipeline(preprocessor)

  #train the models
  logistic_model.fit(X_train, y_train)
  rf_model.fit(X_train, y_train)
  gb_model.fit(X_train, y_train)

  #evaluate the models
  evaluate_model("Logistic train", logistic_model, X_train, y_train)
  # evaluate_model("Logistic validation", logistic_model, X_val, y_val)
  # evaluate_model("RandomForest train", rf_model, X_train, y_train)
  # evaluate_model("RandomForest validation", rf_model, X_val, y_val)
  # evaluate_model("GradientBoosting train", gb_model, X_train, y_train)
  # evaluate_model("GradientBoosting validation", gb_model, X_val, y_val)
  
  

  models = {
    "Logistic": logistic_model,
    "RandomForest": rf_model,
    "GradientBoosting": gb_model
  }

  compare_models(models, X_val, y_val)

  #save the model
  save_model(logistic_model)

if __name__ == "__main__":
  main()
