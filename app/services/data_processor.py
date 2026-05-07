import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import LabelEncoder
import os

class DataProcessor:
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}

    def load_data(self, df: pd.DataFrame = None):
        if df is not None:
            self.df = df.copy()
        elif self.data_path and os.path.exists(self.data_path):
            self.df = pd.read_excel(self.data_path)
        else:
            raise FileNotFoundError("Data file not provided or does not exist.")
            
        # Standardize column names
        cols_map = {c: c.lower() for c in self.df.columns}
        self.df.rename(columns=cols_map, inplace=True)
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        return self.df

    def get_states(self):
        if self.df is not None and 'state' in self.df.columns:
            return self.df['state'].unique().tolist()
        return []

    def process_state_data(self, state: str):
        state_df = self.df[self.df['state'] == state].copy()
        
        processed_dfs = []
        has_category = 'category' in state_df.columns and not state_df['category'].isna().all()
        
        if has_category:
            categories = state_df['category'].dropna().unique()
            for cat in categories:
                cat_df = state_df[state_df['category'] == cat].copy()
                cat_df = self._fill_missing_dates(cat_df)
                cat_df = self._add_time_features(cat_df)
                cat_df = self._add_lag_rolling_features(cat_df)
                processed_dfs.append(cat_df)
            
            final_df = pd.concat(processed_dfs, ignore_index=True)
            
            # Encode category
            le = LabelEncoder()
            final_df['category_encoded'] = le.fit_transform(final_df['category'])
            self.label_encoders[state] = le
            # One-hot encoding as well for models that prefer it
            dummies = pd.get_dummies(final_df['category'], prefix='cat')
            final_df = pd.concat([final_df, dummies], axis=1)
        else:
            final_df = self._fill_missing_dates(state_df)
            final_df = self._add_time_features(final_df)
            final_df = self._add_lag_rolling_features(final_df)
            
        final_df = final_df.sort_values(by=['date']).reset_index(drop=True)
        # Drop rows with NaN from lag features
        final_df = final_df.dropna()
        return final_df, has_category

    def _fill_missing_dates(self, df: pd.DataFrame):
        df = df.set_index('date').sort_index()
        # Aggregate duplicates if any
        df = df.groupby(df.index).first()
        
        # Reindex to daily frequency
        idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df = df.reindex(idx)
        
        # Interpolate target
        if 'total' in df.columns:
            df['total'] = pd.to_numeric(df['total'], errors='coerce')
            df['total'] = df['total'].interpolate(method='linear').bfill().ffill()
            
        # Forward fill state and category
        if 'state' in df.columns:
            df['state'] = df['state'].ffill().bfill()
        if 'category' in df.columns:
            df['category'] = df['category'].ffill().bfill()
            
        df = df.reset_index().rename(columns={'index': 'date'})
        return df

    def _add_time_features(self, df: pd.DataFrame):
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        us_holidays = holidays.US()
        df['holiday_flag'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)
        return df

    def _add_lag_rolling_features(self, df: pd.DataFrame):
        df['lag_1'] = df['total'].shift(1)
        df['lag_7'] = df['total'].shift(7)
        df['lag_30'] = df['total'].shift(30)
        
        df['rolling_mean_7'] = df['total'].shift(1).rolling(window=7).mean()
        df['rolling_std_7'] = df['total'].shift(1).rolling(window=7).std()
        return df
