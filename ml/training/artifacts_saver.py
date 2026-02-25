import json
import joblib
from pathlib import Path

def save_artifacts(preprocessor, best_threshold):

  project_root = Path(__file__).resolve().parent.parent.parent
  model_path = project_root / "src" / "customer_churn_ai" / "models"
  
  model_path.mkdir(parents=True, exist_ok=True)

  print(f"Artifacts saved to {model_path}")



  joblib.dump(preprocessor, model_path / "preprocessor.pkl")

  with open(model_path / "threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f)
