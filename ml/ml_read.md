
# Theory 

## .fit() vs .transform() vs .predict() 

1. fit() -> “learn from data”\
  Computes statistics or patterns from the training data.\
  Examples:
    - Median or mean for missing values (imputer)
    - Mean and standard deviation for scaling (scaler)
    - Categories for one-hot encoding (encoder)
    - Model coefficients for logistic regression

> [!WARNING] 
> Always fit on training data only to avoid data leakage.

```bash
preprocessor.fit(X_train)
# - Stores median, mean, std, and categories internally
# - Prepares transformation rules
``` 


2. .transform() -> “apply learned rules to data”
  Replace missing values using learned median/most frequent\
  Scale numerical features using learned mean/std\
  Encode categorical features using learned categories\


Example:

```bash
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# X_train and X_test are now numeric matrices ready for the model
``` 

> [!WARNING] 
> fit is not called on X_test → prevents leaking future info

3. .predict()
  Applies preprocessing automatically (if in a pipeline)\
  Runs the model on transformed data\
  Returns predictions (e.g., 0/1 or probabilities)\

Example:
```bash
model = Pipeline([
  ("preprocessor", build_preprocessor(num_features, cat_features)),
  ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)      # Learn everything (impute, scale, encode, model)
predictions = model.predict(X_test)  # Preprocess X_test and predict churn
``` 



### Raw Data → Impute → Scale/Encode → Model → Prediction
![](ml_process.png?raw=true)