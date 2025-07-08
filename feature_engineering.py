# feature_engineering.py

import pandas as pd
import numpy as np

def create_gb_features(df, days=4):
    """
    Engineers summary features from early-stage time-series data.
    
    Args:
        df (pd.DataFrame): DataFrame with 'batch_id' and sensor columns.
        days (int): The number of early days to use for features.
        
    Returns:
        pd.DataFrame: A DataFrame with one row of features per batch.
    """
    early_df = df[df['day'] <= days]
    
    features = early_df.groupby('batch_id').agg(
        mean_temp=('temperature', 'mean'),
        std_temp=('temperature', 'std'),
        mean_ph=('ph', 'mean'),
        std_ph=('ph', 'std'),
        mean_do=('dissolved_oxygen', 'mean'),
        std_do=('dissolved_oxygen', 'std')
    )
    return features