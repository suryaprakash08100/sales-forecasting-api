# End-to-End Time Series Forecasting System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
A production-ready, end-to-end time series forecasting system designed to predict the next 8 weeks (56 days) of state-level sales data. The project evaluates multiple state-of-the-art forecasting algorithms (SARIMA, Prophet, XGBoost, LSTM) against historical data, automatically selects the best-performing model based on RMSE, and exposes the generated forecasts via a REST API.

---

## 🎥 Demo Video
> **[Insert Link to Your Demo Video Here]**
*(A short walkthrough of the codebase, training process, and a live demonstration of the Swagger UI fetching state-level predictions.)*

---

## 🏗️ Architecture
The system is designed mimicking a real-world enterprise backend service, featuring a clear separation of concerns:
```text
forecasting-system/
├── app/
│   ├── api/            # API routing and endpoint logic
│   ├── models/         # Pydantic schemas for data validation
│   ├── services/       # Core business logic (Preprocessing, Modeling, Inference)
│   ├── saved_models/   # Serialized best-performing models & tail data
│   └── main.py         # FastAPI application initialization
├── data/               # Raw dataset directory (ignored in version control)
├── Dockerfile          # Containerization instructions
├── requirements.txt    # Project dependencies
└── train.py            # CLI orchestration script for training pipelines
```

---

## 📊 Dataset Description
The system expects an Excel dataset (`.xlsx`) containing historical sales data.
* **Granularity:** Daily frequency.
* **Target Variable:** `Total` (Continuous numerical value representing sales).
* **Features:** `Date` (Datetime), `State` (Categorical), and `Category` (Categorical).

---

## ⚙️ Data Preprocessing & Feature Engineering

### Preprocessing Pipeline (`data_processor.py`)
Handling raw data anomalies is critical for robust time-series forecasting:
* **Continuous Reindexing:** Ensures no missing dates exist in the timeline.
* **Missing Value Imputation:** Interpolates missing `Total` values using linear interpolation, backed by forward and backward filling.
* **Categorical Encoding:** One-hot encodes the `Category` variable dynamically to allow models to capture category-specific signals.

### Feature Engineering
The pipeline constructs rich temporal features to help models identify hidden patterns:
* **Lags:** `lag_1`, `lag_7`, `lag_30` (Captures autoregressive properties).
* **Rolling Statistics:** `rolling_mean_7`, `rolling_std_7` (Captures local momentum and volatility).
* **Calendar Features:** `day_of_week`, `month` (Captures weekly and yearly seasonality).
* **Event Flags:** `holiday_flag` (Utilizes the `holidays` Python package to flag US national holidays).

---

## 🤖 Models Used & Evaluation

### Evaluated Algorithms
1. **SARIMA:** Traditional statistical model capturing autoregression, differencing, and moving averages with seasonality.
2. **Facebook Prophet:** Additive regression model highly robust to missing data and shifts in trend.
3. **XGBoost:** Gradient boosted trees utilizing the engineered lag and rolling features.
4. **LSTM (Deep Learning):** A Recurrent Neural Network built with Keras/TensorFlow to capture deep sequential dependencies.

### Evaluation Metrics
Models are evaluated using a strict **Time-Series Split (Cross-Validation)** to entirely prevent data leakage. The following metrics are computed:
* **RMSE** (Root Mean Squared Error) - *Used as the primary metric for best model selection.*
* **MAE** (Mean Absolute Error)
* **MAPE** (Mean Absolute Percentage Error)

---

## 🚀 API Endpoints & Swagger Docs
The predictions are served via a FastAPI REST application.

* **Endpoint:** `GET /api/v1/forecast/{state}`
* **Query Parameters:** `days` (integer, default=56)
* **Response:** JSON payload containing the exact model used and the day-by-day predictions.

**Swagger Documentation:** Once the server is running, interactive API documentation is available at `http://127.0.0.1:8000/docs`.

---

## 💻 How to Run Locally

### 1. Setup Environment
```bash
# Clone the repository and navigate to the directory
cd forecasting-system

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Models
Ensure your data file is located in the `data/` directory.
```bash
python train.py --data_path data/your_dataset.xlsx
```
*This script will process the data, train all 4 models, compare them, and save the best artifacts to `app/saved_models/`.*

### 3. Run the API Server
```bash
uvicorn app.main:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` in your browser to test the endpoints.

---

## 🐳 Docker Usage
To containerize and run the application using Docker:

```bash
# Build the Docker image
docker build -t forecasting-api .

# Run the container
docker run -p 8000:8000 forecasting-api
```
The API will be instantly accessible at `http://localhost:8000/docs`.

---

## 🔮 Future Improvements
* **Hyperparameter Tuning:** Implement Optuna or GridSearch to fine-tune XGBoost and LSTM architectures automatically.
* **Global Modeling:** Train a single global model capable of zero-shot forecasting across all states concurrently rather than independent models.
* **Caching Layer:** Implement Redis to cache frequent API forecast requests and reduce inference latency.
* **CI/CD Integration:** Add GitHub Actions for automated unit testing and container deployment.
