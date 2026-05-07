import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, train_df, target='total', exog_cols=None):
        endog = train_df[target]
        exog = train_df[exog_cols] if exog_cols else None
        model = SARIMAX(endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
        self.model_fit = model.fit(disp=False)

    def predict(self, test_df, exog_cols=None):
        exog = test_df[exog_cols] if exog_cols else None
        return self.model_fit.forecast(steps=len(test_df), exog=exog).values

class ProphetModel:
    def __init__(self):
        self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.regressors = []

    def fit(self, train_df, target='total', exog_cols=None):
        df = train_df.rename(columns={'date': 'ds', target: 'y'})
        self.regressors = exog_cols if exog_cols else []
        for col in self.regressors:
            self.model.add_regressor(col)
        self.model.fit(df)

    def predict(self, test_df, exog_cols=None):
        df = test_df.rename(columns={'date': 'ds'})
        forecast = self.model.predict(df)
        return forecast['yhat'].values

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    def fit(self, train_df, target='total', exog_cols=None):
        X = train_df[exog_cols]
        y = train_df[target]
        self.model.fit(X, y)

    def predict(self, test_df, exog_cols=None):
        X = test_df[exog_cols]
        return self.model.predict(X)

class LSTMTimeSeriesModel:
    def __init__(self, lookback=7):
        self.lookback = lookback
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _prepare_data(self, df, target, exog_cols):
        X = df[exog_cols].values
        y = df[target].values.reshape(-1, 1)
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-self.lookback:i, :])
            y_seq.append(y_scaled[i])
            
        return np.array(X_seq), np.array(y_seq)

    def fit(self, train_df, target='total', exog_cols=None):
        if len(train_df) <= self.lookback:
            raise ValueError("Training data length must be greater than lookback period.")
            
        X_train, y_train = self._prepare_data(train_df, target, exog_cols)
        
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        
        es = EarlyStopping(monitor='loss', patience=5, verbose=0)
        self.model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, callbacks=[es])

    def predict(self, test_df, exog_cols=None):
        # For simplicity in evaluation, we just transform test_df (ignoring the lookback gap for pure evaluation metric)
        # In a strict setting, we'd need history. Here we just take sliding windows within test_df.
        # To match lengths, we'll pad the beginning with the first available values or we can use the original X
        
        X = test_df[exog_cols].values
        X_scaled = self.scaler_X.transform(X)
        
        X_seq = []
        # Pad initial lookback with the first row
        padded_X = np.vstack([np.repeat(X_scaled[0:1], self.lookback, axis=0), X_scaled])
        
        for i in range(self.lookback, len(padded_X)):
            X_seq.append(padded_X[i-self.lookback:i, :])
            
        X_seq = np.array(X_seq)
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        return self.scaler_y.inverse_transform(y_pred_scaled).flatten()

def get_model(model_name: str):
    models = {
        'SARIMA': SARIMAModel(),
        'Prophet': ProphetModel(),
        'XGBoost': XGBoostModel(),
        'LSTM': LSTMTimeSeriesModel()
    }
    if model_name in models:
        return models[model_name]
    raise ValueError(f"Unknown model: {model_name}")
