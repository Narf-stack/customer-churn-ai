import numpy as np
from backend.app.ml import model, preprocessor, threshold
from backend.app.schemas import ChurnRequest
from ml.training.threshold import apply_threshold

def predict_churn(request: ChurnRequest):
  data = np.array([[
    request.tenure,
    request.monthly_charges,
    request.total_charges
  ]])

  processed = preprocessor.transform(data)

  proba = model.predict_proba(processed)[:, 1][0]
  prediction = apply_threshold(np.array([proba]), threshold)[0]

  return {
    "churn_probability": float(proba),
    "churn_prediction": int(prediction)
  }

