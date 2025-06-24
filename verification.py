import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your multi-device prediction CSV
df = pd.read_csv("all_devices_predictions.csv", parse_dates=['ds'])

# SMAPE function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Compute overall metrics
rmse = np.sqrt(mean_squared_error(df['y'], df['yhat']))
mae = mean_absolute_error(df['y'], df['yhat'])
smape_val = smape(df['y'].values, df['yhat'].values)
accuracy = 100 - smape_val

print("\n📊 Overall Multi-Device Prophet Model Performance")
print("----------------------------------------")
print(f"Total Data Points : {len(df)}")
print(f"SMAPE             : {smape_val:.2f}%")
print(f"Accuracy          : {accuracy:.2f}%")
print(f"RMSE              : {rmse:.4f}")
print(f"MAE               : {mae:.4f}")
print("----------------------------------------")
