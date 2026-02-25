from pydantic import BaseModel

class ChurnRequest(BaseModel):
  tenure: int
  monthly_charges: float
  total_charges: float
  contract: str
  payment_method: str


class ChurnResponse(BaseModel):
  churn_probability: float
  churn_prediction: int
