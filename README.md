# Time Series Forecasting System

Production-ready end-to-end time series forecasting system.

## Features
- Forecast next 8 weeks of sales for each state
- Handles missing dates and values
- Evaluates SARIMA, Prophet, XGBoost, and LSTM automatically
- Time-series split cross-validation
- Exposes predictions via FastAPI REST endpoints

## Setup

1. Place your Excel data file in the `data/` directory. It should have columns: `State`, `Date`, `Total`, `Category`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the models on your dataset and automatically save the best models:
```bash
python train.py --data_path data/your_data.xlsx
```

## Running the API

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
View Swagger documentation at: `http://localhost:8000/docs`
