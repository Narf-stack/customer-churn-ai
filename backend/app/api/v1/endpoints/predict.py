from fastapi import APIRouter
from backend.app.schemas import ChurnRequest, ChurnResponse
from backend.app.services.prediction_service import predict_churn

router = APIRouter()

@router.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
  return predict_churn(request)
