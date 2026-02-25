import pandas as pd
import numpy as np
from backend.app.core.model_registry import registry
from backend.app.schemas import ChurnRequest
from ml.training.threshold import apply_threshold

def predict_churn(request: ChurnRequest):
  data = pd.DataFrame([{
    "gender": request.gender,
    "SeniorCitizen": request.SeniorCitizen,
    "Partner": request.Partner,
    "Dependents": request.Dependents,
    "tenure": request.tenure,
    "PhoneService": request.PhoneService,
    "MultipleLines": request.MultipleLines,
    "InternetService": request.InternetService,
    "OnlineSecurity": request.OnlineSecurity,
    "OnlineBackup": request.OnlineBackup,
    "DeviceProtection": request.DeviceProtection,
    "TechSupport": request.TechSupport,
    "StreamingTV": request.StreamingTV,
    "StreamingMovies": request.StreamingMovies,
    "Contract": request.Contract,
    "PaperlessBilling": request.PaperlessBilling,
    "PaymentMethod": request.PaymentMethod,
    "MonthlyCharges": request.MonthlyCharges,
    "TotalCharges": request.TotalCharges
  }])

  # processed = registry.preprocessor.transform(data)
  # breakpoint()
  proba = registry.model.predict_proba(data)[:, 1][0]
  prediction = apply_threshold(np.array([proba]), registry.threshold)[0]

  return {
    "churn_probability": float(proba),
    "churn_prediction": int(prediction)
  }

