import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm

# SMAPE function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Load dataset
df = pd.read_csv("12lakh_predictive_maintenance_dataset.csv", parse_dates=['date'])

# Prepare list to store all results
all_results = []

# List of unique devices (limit to top N if needed for speed)
devices = df['device'].unique()

for device in tqdm(devices, desc="Processing devices"):
    df_device = df[df['device'] == device].copy()

    # Aggregate daily data
    agg_df = df_device.groupby('date').agg({
        'failure': 'sum',
        'metric1': 'mean',
        'metric2': 'mean',
        'metric3': 'mean',
        'metric4': 'mean',
        'metric5': 'mean',
        'metric6': 'mean',
        'metric7': 'mean',
        'metric8': 'mean',
        'metric9': 'mean',
    }).reset_index()

    if len(agg_df) < 60:  # skip if not enough data
        continue

    # Smooth failure
    agg_df['failure'] = agg_df['failure'].rolling(window=3, min_periods=1).mean()
    agg_df.rename(columns={'date': 'ds', 'failure': 'y'}, inplace=True)

    # Split train/test
    train_df = agg_df[:-30]
    test_df = agg_df[-30:]

    try:
        # Build and train model
        model = Prophet()
        regressors = [f'metric{i}' for i in range(1, 10)]
        for reg in regressors:
            model.add_regressor(reg)

        model.fit(train_df[['ds', 'y'] + regressors])

        # Predict
        future = agg_df[['ds'] + regressors].copy()
        forecast = model.predict(future)

        # Merge actual & predicted
        merged = pd.merge(forecast[['ds', 'yhat']], test_df[['ds', 'y']], on='ds')
        merged['device'] = device

        all_results.append(merged)

    except Exception as e:
        print(f"Skipped {device}: {e}")
        continue

# Combine all predictions
final_df = pd.concat(all_results, ignore_index=True)

# Save to CSV
final_df.to_csv("all_devices_predictions.csv", index=False)

print("✅ Forecasting complete. Results saved to 'all_devices_predictions.csv'.")
