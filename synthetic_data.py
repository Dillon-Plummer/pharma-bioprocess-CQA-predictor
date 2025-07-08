# synthetic_data.py

import pandas as pd
import numpy as np

def generate_synthetic_data():
    """
    Generates and returns a synthetic cell culture dataset as a pandas DataFrame.
    The data includes embedded trends for models to learn.
    """
    all_data = []
    num_batches = 20
    days = 14
    time_points_per_day = 24

    for i in range(num_batches):
        batch_id = f"batch_{i+1:02d}"
        
        # Assign a profile to each batch to create clear patterns
        if i < 8:
            profile = "Good"  # High Titer
        elif i < 14:
            profile = "Okay"  # Medium Titer (Temp issues)
        else:
            profile = "Poor"  # Low Titer (pH crash)

        # Initialize parameters based on profile
        if profile == "Good":
            temp_mean, temp_std = 37.0, 0.05
            ph_mean, ph_std = 7.0, 0.03
            final_titer = np.random.uniform(0.85, 0.95)
        elif profile == "Okay":
            temp_mean, temp_std = 37.0, 0.2
            ph_mean, ph_std = 7.0, 0.04
            final_titer = np.random.uniform(0.55, 0.65)
        else: # Poor
            temp_mean, temp_std = 37.0, 0.06
            ph_mean, ph_std = 6.9, 0.05
            final_titer = np.random.uniform(0.30, 0.40)

        # Generate time series data for the batch
        for day in range(days):
            for hour in range(time_points_per_day):
                current_time = day + hour / time_points_per_day
                
                temp = np.random.normal(temp_mean, temp_std)
                ph = np.random.normal(ph_mean, ph_std)
                
                # Apply the pH crash trend for "Poor" batches
                if profile == "Poor" and day >= 2:
                    ph -= (day - 1.5) * 0.05 

                # Dissolved Oxygen trend
                do = 90 * np.exp(-0.2 * current_time) + np.random.uniform(-2, 2)

                all_data.append({
                    "batch_id": batch_id,
                    "day": day,
                    "time": current_time,
                    "temperature": temp,
                    "ph": ph,
                    "dissolved_oxygen": do,
                    "titer": final_titer 
                })

    return pd.DataFrame(all_data)

