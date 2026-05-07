import os
import json
import joblib
import pandas as pd
import argparse
from app.services.data_processor import DataProcessor
from app.services.models_factory import get_model
from app.services.evaluator import Evaluator

def save_model(model_obj, model_name, state, metrics, exog_cols, df_tail, save_dir='app/saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    state_safe = str(state).replace(' ', '_').replace('/', '_')
    
    metadata = {
        'state': state,
        'model_name': model_name,
        'metrics': metrics,
        'exog_cols': exog_cols
    }
    
    with open(f'{save_dir}/{state_safe}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    df_tail.to_csv(f'{save_dir}/{state_safe}_tail.csv', index=False)
        
    if model_name == 'LSTM':
        model_obj.model.save(f'{save_dir}/{state_safe}_model.keras')
        # Also save scalers
        joblib.dump(model_obj.scaler_X, f'{save_dir}/{state_safe}_scaler_X.pkl')
        joblib.dump(model_obj.scaler_y, f'{save_dir}/{state_safe}_scaler_y.pkl')
    else:
        # Some objects like SARIMAX might need specific saving, but joblib works for most
        # For Prophet, we can just joblib the Prophet object
        joblib.dump(model_obj.model if hasattr(model_obj, 'model') else model_obj.model_fit, 
                    f'{save_dir}/{state_safe}_model.pkl')

def train_all(data_path: str):
    processor = DataProcessor(data_path)
    processor.load_data()
    states = processor.get_states()
    
    evaluator = Evaluator(n_splits=3)
    model_names = ['SARIMA', 'Prophet', 'XGBoost', 'LSTM']
    
    for state in states:
        print(f"\nProcessing state: {state}")
        df, has_category = processor.process_state_data(state)
        
        # Determine features
        base_features = [
            'lag_1', 'lag_7', 'lag_30', 
            'rolling_mean_7', 'rolling_std_7', 
            'day_of_week', 'month', 'holiday_flag'
        ]
        
        if has_category:
            # Add one-hot encoded category columns
            cat_cols = [c for c in df.columns if c.startswith('cat_')]
            exog_cols = base_features + cat_cols
        else:
            exog_cols = base_features
            
        # Ensure columns exist (SARIMA/Prophet might not need lags, but we pass them for uniform interface)
        results = {}
        for name in model_names:
            print(f"  Evaluating {name}...")
            mae, rmse, mape = evaluator.evaluate_model(name, df, target='total', exog_cols=exog_cols)
            results[name] = {'mae': mae, 'rmse': rmse, 'mape': mape}
            print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}")
            
        best_model_name = evaluator.select_best_model(results)
        print(f"  Best model for {state}: {best_model_name}")
        
        # Retrain on full data
        best_model = get_model(best_model_name)
        best_model.fit(df, target='total', exog_cols=exog_cols)
        
        # Save best model and the last 30 rows per category (or just last 30 rows if no category) for lag features
        if has_category:
            df_tail = df.groupby('category').tail(30)
        else:
            df_tail = df.tail(30)
            
        save_model(best_model, best_model_name, state, results[best_model_name], exog_cols, df_tail)
        print(f"  Model saved for {state}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument('--data_path', type=str, required=True, help="Path to historical Excel data")
    args = parser.parse_args()
    
    train_all(args.data_path)
