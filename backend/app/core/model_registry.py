from typing import Optional
from sklearn.base import BaseEstimator

class ModelRegistry:
  model: Optional[BaseEstimator] = None
  preprocessor: Optional[BaseEstimator] = None
  threshold: Optional[float] = None


registry = ModelRegistry()
