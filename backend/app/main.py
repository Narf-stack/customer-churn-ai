from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from backend.app.core import registry
from backend.app.api.v1.endpoints import predict
from backend.app.ml import model, preprocessor, threshold
from backend.app.api.v1.endpoints import feature_importance

@asynccontextmanager
async def lifespan(app: FastAPI):
  registry.model = model
  registry.preprocessor = preprocessor
  registry.threshold = threshold

  print("Model artifacts loaded successfully.")

  yield

app = FastAPI(
  title="Customer Churn API",
  lifespan=lifespan
)

app.include_router(predict.router, prefix="/api/v1")
app.include_router(feature_importance.router, prefix="/api/v1")
