import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as forecast_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Time Series Forecasting API",
    description="API for forecasting future sales based on historical data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Time Series Forecasting API. Visit /docs for documentation."}
