import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# SMAPE function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Load dataset
df = pd.read_csv("predictive_maintenance_dataset.csv", parse_dates=['date'])

# Store results
all_results = []

# Loop through devices
devices = df['device'].unique()
for device in devices:
    df_device = df[df['device'] == device]
    if len(df_device['date'].unique()) < 60:
        continue

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

    agg_df['failure'] = agg_df['failure'].rolling(window=3, min_periods=1).mean()
    agg_df.rename(columns={'date': 'ds', 'failure': 'y'}, inplace=True)

    if agg_df['y'].sum() == 0 or len(agg_df) < 60:
        continue

    train_df = agg_df[:-30]
    test_df = agg_df[-30:]

    model = Prophet()
    regressors = [f'metric{i}' for i in range(1, 10)]
    for reg in regressors:
        model.add_regressor(reg)

    try:
        model.fit(train_df[['ds', 'y'] + regressors])
        future = agg_df[['ds'] + regressors].copy()
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']].merge(agg_df[['ds', 'y']], on='ds')
        forecast['device'] = device
        all_results.append(forecast)
    except:
        continue

# Combine all results
result_df = pd.concat(all_results)

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(result_df['y'], result_df['yhat']))
mae = mean_absolute_error(result_df['y'], result_df['yhat'])
smape_val = smape(result_df['y'].values, result_df['yhat'].values)
accuracy = 100 - smape_val

# Early failure prediction logic
failure_threshold = 0.001  # Tune as needed
lead_days = 5
failure_events = result_df[result_df['y'] > failure_threshold]['ds'].unique()
early_warnings_count = 0

for failure_date in failure_events:
    prediction_window = pd.date_range(end=failure_date - pd.Timedelta(days=1), periods=lead_days)
    window_preds = result_df[result_df['ds'].isin(prediction_window)]
    if any(window_preds['yhat'] > failure_threshold):
        early_warnings_count += 1

total_failures = len(failure_events)
early_detection_rate = (early_warnings_count / total_failures * 100) if total_failures > 0 else 0


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Binarize actual and predicted values using failure threshold
result_df['actual_failure'] = result_df['y'] > failure_threshold
result_df['predicted_failure'] = result_df['yhat'] > failure_threshold

# Compute confusion matrix
cm = confusion_matrix(result_df['actual_failure'], result_df['predicted_failure'])
labels = ['No Failure', 'Failure']

# Display confusion matrix
print("\n🧩 Confusion Matrix (Failure Prediction)")
print("----------------------------------------")
print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

# Optional: Plot confusion matrix
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Failure Prediction Confusion Matrix")
plt.grid(False)
plt.show()


# Summary Output
print("\n📊 Overall Multi-Device Prophet Model Performance")
print("----------------------------------------")
print(f"Total Data Points : {len(result_df)}")
print(f"SMAPE             : {smape_val:.2f}%")
print(f"Accuracy          : {accuracy:.2f}%")
print(f"RMSE              : {rmse:.4f}")
print(f"MAE               : {mae:.4f}")
print("----------------------------------------")

print("\n🔍 Early Failure Detection")
print("----------------------------------------")
print(f"Failure threshold   : {failure_threshold}")
print(f"Look-ahead window   : {lead_days} days")
print(f"Actual Failures     : {total_failures}")
print(f"Early Predictions   : {early_warnings_count}")
print(f"Early Detection Rate: {early_detection_rate:.2f}%")
print("----------------------------------------")