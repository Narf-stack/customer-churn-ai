from fastapi import FastAPI
from .api.v1.endpoints import predict

app = FastAPI(title="Customer Churn API")

app.include_router(predict.router, prefix="/api/v1")
