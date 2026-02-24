import joblib
from pathlib import Path

def save_model(model):
  model_path = Path("../src/customer_churn_ai/model")
  model_path.mkdir(parents=True, exist_ok=True)

  joblib.dump(model, model_path / "churn_pipeline.pkl")
