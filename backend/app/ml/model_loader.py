import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent



MODEL_PATH = BASE_DIR / "src" / "customer_churn_ai" / "models" / "churn_pipeline.pkl"
PREPROCESSOR_PATH = BASE_DIR / "src" / "customer_churn_ai" / "models" / "preprocessor.pkl"
THRESHOLD_PATH = BASE_DIR / "src" / "customer_churn_ai" / "models" / "threshold.json"
METADATA_PATH = BASE_DIR / "src" / "customer_churn_ai" / "models" / "model_metadata.json"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(METADATA_PATH) as f:
  metadata = json.load(f)

with open(METADATA_PATH) as f:
  metadata = json.load(f)

with open(THRESHOLD_PATH) as f:
  threshold = json.load(f)["threshold"]
