from fastapi import APIRouter
from backend.app.services import get_feature_importance

router = APIRouter()

@router.get("/feature-importance")
def feature_importance():
  return get_feature_importance()
