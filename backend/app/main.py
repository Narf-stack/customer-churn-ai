from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from backend.app.core import registry
from backend.app.api.v1.endpoints import predict
from backend.app.ml import model, preprocessor, threshold, metadata
from backend.app.api.v1.endpoints import feature_importance
from backend.app.api.v1.endpoints import model_metadata

@asynccontextmanager
async def lifespan(app: FastAPI):
  registry.model = model_metadata
  registry.preprocessor = preprocessor
  registry.threshold = threshold
  registry.metadata = metadata


  print("Model artifacts loaded successfully.")

  yield

app = FastAPI(
  title="Customer Churn API",
  lifespan=lifespan
)

app.include_router(predict.router, prefix="/api/v1")
app.include_router(feature_importance.router, prefix="/api/v1")
app.include_router(model_metadata.router, prefix="/api/v1")
