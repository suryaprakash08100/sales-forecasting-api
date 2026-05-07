from fastapi import APIRouter, HTTPException
from app.models.schemas import ForecastResponse, ForecastPrediction
from app.services.inference import InferenceService

router = APIRouter()
inference_service = InferenceService()

@router.get("/forecast/{state}", response_model=ForecastResponse)
def get_forecast(state: str, days: int = 56):
    try:
        preds_df, model_name = inference_service.generate_forecast(state, days)
        
        predictions = []
        for _, row in preds_df.iterrows():
            predictions.append(ForecastPrediction(
                date=row['date'],
                predicted_total=row['predicted_total'],
                category=row['category'] if row['category'] else None
            ))
            
        return ForecastResponse(
            state=state,
            model_used=model_name,
            forecast_days=days,
            predictions=predictions
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
