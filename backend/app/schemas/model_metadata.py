from pydantic import BaseModel


class ModelMetadataResponse(BaseModel):
  model_name: str = "Customer Churn Predictor"
  model_type: str
  version: str = "1.0.0"
  training_date: str
  training_samples: int
  features_count: int
  class_labels: list
  threshold: float
  metrics: dict
  environment: dict

