import os
import json
import joblib
import pandas as pd
import numpy as np
import holidays
from app.services.models_factory import get_model, SARIMAModel, ProphetModel, XGBoostModel, LSTMTimeSeriesModel
from tensorflow.keras.models import load_model

class InferenceService:
    def __init__(self, saved_models_dir='app/saved_models'):
        self.saved_models_dir = saved_models_dir

    def load_metadata(self, state_safe: str):
        path = f'{self.saved_models_dir}/{state_safe}_metadata.json'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model metadata not found for state {state_safe}")
        with open(path, 'r') as f:
            return json.load(f)

    def load_model_obj(self, state_safe: str, model_name: str):
        if model_name == 'LSTM':
            model_wrapper = LSTMTimeSeriesModel()
            model_wrapper.model = load_model(f'{self.saved_models_dir}/{state_safe}_model.keras')
            model_wrapper.scaler_X = joblib.load(f'{self.saved_models_dir}/{state_safe}_scaler_X.pkl')
            model_wrapper.scaler_y = joblib.load(f'{self.saved_models_dir}/{state_safe}_scaler_y.pkl')
            return model_wrapper
        else:
            model_artifact = joblib.load(f'{self.saved_models_dir}/{state_safe}_model.pkl')
            if model_name == 'SARIMA':
                wrapper = SARIMAModel()
                wrapper.model_fit = model_artifact
                return wrapper
            elif model_name == 'Prophet':
                wrapper = ProphetModel()
                wrapper.model = model_artifact
                return wrapper
            elif model_name == 'XGBoost':
                wrapper = XGBoostModel()
                wrapper.model = model_artifact
                return wrapper
        raise ValueError(f"Unknown model {model_name}")

    def generate_forecast(self, state: str, days: int = 56):
        state_safe = str(state).replace(' ', '_').replace('/', '_')
        metadata = self.load_metadata(state_safe)
        model_name = metadata['model_name']
        exog_cols = metadata['exog_cols']
        
        model_wrapper = self.load_model_obj(state_safe, model_name)
        
        # Load tail data
        df_tail = pd.read_csv(f'{self.saved_models_dir}/{state_safe}_tail.csv')
        df_tail['date'] = pd.to_datetime(df_tail['date'])
        
        has_category = 'category' in df_tail.columns
        
        all_predictions = []
        
        if has_category:
            categories = df_tail['category'].unique()
            for cat in categories:
                cat_tail = df_tail[df_tail['category'] == cat].copy()
                cat_preds = self._forecast_iterative(model_wrapper, cat_tail, days, exog_cols, model_name)
                cat_preds['category'] = cat
                all_predictions.append(cat_preds)
            final_preds = pd.concat(all_predictions, ignore_index=True)
        else:
            final_preds = self._forecast_iterative(model_wrapper, df_tail, days, exog_cols, model_name)
            final_preds['category'] = None
            
        return final_preds, model_name

    def _forecast_iterative(self, model_wrapper, df_tail, days, exog_cols, model_name):
        df_tail = df_tail.sort_values('date').reset_index(drop=True)
        us_holidays = holidays.US()
        
        predictions = []
        current_df = df_tail.copy()
        
        # Need to know which features to compute
        compute_lags = any('lag_' in col or 'rolling_' in col for col in exog_cols)
        
        last_date = current_df['date'].max()
        
        for i in range(1, days + 1):
            next_date = last_date + pd.Timedelta(days=i)
            
            # Create next row template
            next_row = {
                'date': next_date,
                'day_of_week': next_date.dayofweek,
                'month': next_date.month,
                'holiday_flag': 1 if next_date in us_holidays else 0
            }
            
            # Carry over one-hot encoding columns
            for col in exog_cols:
                if col.startswith('cat_') and col in current_df.columns:
                    next_row[col] = current_df.iloc[-1][col]
                    
            if compute_lags:
                next_row['lag_1'] = current_df.iloc[-1]['total']
                next_row['lag_7'] = current_df.iloc[-7]['total'] if len(current_df) >= 7 else current_df.iloc[-1]['total']
                next_row['lag_30'] = current_df.iloc[-30]['total'] if len(current_df) >= 30 else current_df.iloc[0]['total']
                
                # Rolling stats over last 7 days of known 'total'
                last_7 = current_df['total'].tail(7)
                next_row['rolling_mean_7'] = last_7.mean()
                next_row['rolling_std_7'] = last_7.std()
                if pd.isna(next_row['rolling_std_7']):
                    next_row['rolling_std_7'] = 0.0

            next_df = pd.DataFrame([next_row])
            
            # Some models don't need exog for prediction if they are autoregressive on target, 
            # but we pass it anyway to match interface
            if model_name in ['SARIMA', 'Prophet']:
                # For SARIMA/Prophet, we might just pass dates or exog
                pred_val = model_wrapper.predict(next_df, exog_cols=exog_cols)[0]
            elif model_name == 'LSTM':
                # LSTM needs history window for prediction, so we pass the recent history
                # This requires a slightly different approach or padding, but we designed LSTM predict
                # to take the test_df. Wait, LSTM `predict` currently expects sliding windows. 
                # To be robust, we pass the lookback window + new row
                temp_df = pd.concat([current_df.tail(model_wrapper.lookback), next_df], ignore_index=True)
                pred_val = model_wrapper.predict(temp_df, exog_cols=exog_cols)[-1]
            else: # XGBoost
                pred_val = model_wrapper.predict(next_df, exog_cols=exog_cols)[0]
                
            pred_val = max(0, pred_val) # No negative sales
            next_row['total'] = pred_val
            
            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'predicted_total': float(pred_val)
            })
            
            # Append for next iteration
            current_df = pd.concat([current_df, pd.DataFrame([next_row])], ignore_index=True)
            
        return pd.DataFrame(predictions)
