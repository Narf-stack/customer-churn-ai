import joblib
from pathlib import Path

def save_model(model):

  project_root = Path(__file__).resolve().parent.parent.parent
  model_path = project_root / "src" / "customer_churn_ai" / "models"
  
  model_path.mkdir(parents=True, exist_ok=True)

  joblib.dump(model, model_path / "churn_pipeline.pkl")
  print(f"Model saved to {model_path / 'churn_pipeline.pkl'}")
