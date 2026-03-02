import numpy as np
from backend.app.ml import model
from fastapi import HTTPException
from sklearn.pipeline import Pipeline

def get_feature_importance():
  try:
    if not model:
      raise HTTPException(status_code=500, detail="Model not loaded")
    
    if isinstance(model, Pipeline):
      final_estimator = model.steps[-1][1]
    else:
      final_estimator = model

    model_type = type(final_estimator).__name__

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # Get feature names after preprocessing (important!)
    feature_names = preprocessor.get_feature_names_out()

    # Get coefficients from logistic regression
    coefficients = classifier.coef_[0]

    importance_dict = dict(zip(feature_names, coefficients))

    # Sort by absolute importance
    sorted_importance = dict(
      sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
    "model_type": model_type,
    "feature_importance": sorted_importance
    }
 
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
