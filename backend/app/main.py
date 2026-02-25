from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
from backend.app.core import registry
from backend.app.api.v1.endpoints import predict
from backend.app.ml import model, preprocessor, threshold

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
