from backend.app.schemas import ModelMetadataResponse
from fastapi import APIRouter
from backend.app.core import registry


router = APIRouter()

@router.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata():
  return registry.metadata