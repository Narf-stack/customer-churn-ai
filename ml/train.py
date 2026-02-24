# file for orchestration only



from training import *


df = load_data("../data/telco_customer_churn.csv")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

numerical_features, categorical_features = get_feature_types(X_train)
preprocessor = build_preprocessor(numerical_features, categorical_features)
pipeline = build_pipeline(preprocessor)
pipeline.fit(X_train, y_train)
