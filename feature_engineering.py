# feature_engineering.py
import pandas as pd

def create_tabular_features(df_input, is_training=True):
    """
    Engineers summary features from time-series data.

    Args:
        df_input (pd.DataFrame): The input data.
        is_training (bool): If True, groups by 'batch_id'. 
                            If False, processes a single batch.

    Returns:
        pd.DataFrame: A DataFrame with one row of features per batch.
    """
    if is_training:
        # For training: process all batches in the dataframe
        early_df = df_input[df_input['day'] <= 4]
        features = early_df.groupby('batch_id').agg(
            mean_temp=('temperature', 'mean'),
            std_temp=('temperature', 'std'),
            mean_ph=('ph', 'mean'),
            std_ph=('ph', 'std'),
            mean_do=('dissolved_oxygen', 'mean'),
            std_do=('dissolved_oxygen', 'std')
        )
        return features
    else:
        # For prediction: process a single batch dataframe
        return pd.DataFrame({
            'mean_temp': [df_input['temperature'].mean()],
            'std_temp': [df_input['temperature'].std()],
            'mean_ph': [df_input['ph'].mean()],
            'std_ph': [df_input['ph'].std()],
            'mean_do': [df_input['dissolved_oxygen'].mean()],
            'std_do': [df_input['dissolved_oxygen'].std()]
        })
