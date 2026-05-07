from pydantic import BaseModel
from typing import List

class ForecastPrediction(BaseModel):
    date: str
    predicted_total: float
    category: str = None

class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_days: int
    predictions: List[ForecastPrediction]
