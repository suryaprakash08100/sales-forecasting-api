from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from app.services.models_factory import get_model

class Evaluator:
    def __init__(self, n_splits=3):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def evaluate_model(self, model_name, df, target='total', exog_cols=None):
        maes, rmses, mapes = [], [], []
        
        for train_index, test_index in self.tscv.split(df):
            model = get_model(model_name)
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            
            try:
                model.fit(train_df, target=target, exog_cols=exog_cols)
                preds = model.predict(test_df, exog_cols=exog_cols)
                
                y_true = test_df[target].values
                # Avoid negative values if models produce them
                preds = np.maximum(preds, 0)
                
                maes.append(mean_absolute_error(y_true, preds))
                rmses.append(np.sqrt(mean_squared_error(y_true, preds)))
                mapes.append(mean_absolute_percentage_error(y_true, preds))
            except Exception as e:
                # If a model fails for a split (e.g. SARIMA convergence issues)
                print(f"Warning: Evaluation failed for a split. Error: {e}")
                continue
                
        if not maes:
            return float('inf'), float('inf'), float('inf')
            
        return np.mean(maes), np.mean(rmses), np.mean(mapes)

    def select_best_model(self, results):
        # results is a dict: { 'model_name': { 'mae': val, 'rmse': val, 'mape': val } }
        best_model_name = None
        best_rmse = float('inf')
        
        for name, metrics in results.items():
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model_name = name
                
        return best_model_name
