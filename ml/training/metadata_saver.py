import json
from pathlib import Path

def save_metadata(metadata):

  project_root = Path(__file__).resolve().parent.parent.parent
  metadata_path = project_root / "src" / "customer_churn_ai" / "models"
  
  metadata_path.mkdir(parents=True, exist_ok=True)


  with open(metadata_path / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
    
  print(f"Metadata saved to {metadata_path / 'model_metadata.json'}")
