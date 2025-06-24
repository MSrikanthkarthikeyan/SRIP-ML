import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# SMAPE function
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Load dataset
df = pd.read_csv("12lakh_predictive_maintenance_dataset.csv", parse_dates=['date'])

all_results = []

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
    except Exception as e:
        print(f"Model failed for device {device}: {e}")
        continue

result_df = pd.concat(all_results)

# === Overall regression metrics ===
rmse = np.sqrt(mean_squared_error(result_df['y'], result_df['yhat']))
mae = mean_absolute_error(result_df['y'], result_df['yhat'])
smape_val = smape(result_df['y'].values, result_df['yhat'].values)
approx_accuracy = 100 - smape_val

# === Binary classification ===
failure_threshold = 0.001
result_df['actual_failure'] = (result_df['y'] > failure_threshold).astype(int)
result_df['predicted_failure'] = (result_df['yhat'] > failure_threshold).astype(int)

precision = precision_score(result_df['actual_failure'], result_df['predicted_failure'], zero_division=0)
recall = recall_score(result_df['actual_failure'], result_df['predicted_failure'], zero_division=0)
f1 = f1_score(result_df['actual_failure'], result_df['predicted_failure'], zero_division=0)

# === Confusion Matrix ===
cm = confusion_matrix(result_df['actual_failure'], result_df['predicted_failure'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Failure Prediction")
plt.grid(False)
plt.show()

# === Early detection logic ===
lead_days = 5
failure_events = result_df[result_df['actual_failure'] == 1]['ds'].unique()
early_warnings_count = 0

for failure_date in failure_events:
    prediction_window = pd.date_range(end=failure_date - pd.Timedelta(days=1), periods=lead_days)
    early_preds = result_df[(result_df['ds'].isin(prediction_window)) & (result_df['predicted_failure'] == 1)]
    if not early_preds.empty:
        early_warnings_count += 1

total_failures = len(failure_events)
early_detection_rate = (early_warnings_count / total_failures * 100) if total_failures > 0 else 0

# === Report ===
print("\n📊 Overall Multi-Device Prophet Model Performance")
print("-------------------------------------------------")
print(f"Total Data Points    : {len(result_df)}")
print(f"RMSE                 : {rmse:.4f}")
print(f"MAE                  : {mae:.4f}")
print(f"SMAPE                : {smape_val:.2f}%")
print(f"Approx Accuracy      : {approx_accuracy:.2f}%")
print("-------------------------------------------------")
print(f"Precision (Failures) : {precision:.2f}")
print(f"Recall    (Failures) : {recall:.2f}")
print(f"F1 Score             : {f1:.2f}")

print("\n🔍 Early Failure Detection")
print("-------------------------------------------------")
print(f"Failure threshold     : {failure_threshold}")
print(f"Look-ahead window     : {lead_days} days")
print(f"Total Failures        : {total_failures}")
print(f"Early Predictions     : {early_warnings_count}")
print(f"Early Detection Rate  : {early_detection_rate:.2f}%")
print("-------------------------------------------------")

# === Visualization of average predictions vs actuals ===
agg_daily = result_df.groupby('ds').agg({
    'y': 'mean',
    'yhat': 'mean'
}).reset_index()

failure_days = agg_daily[agg_daily['y'] > failure_threshold]['ds']

plt.figure(figsize=(15, 7))
plt.plot(agg_daily['ds'], agg_daily['y'], label='Average Actual Failure (smoothed)', color='red', linewidth=2)
plt.plot(agg_daily['ds'], agg_daily['yhat'], label='Average Predicted Failure', color='blue', linestyle='--')

plt.scatter(failure_days, agg_daily[agg_daily['ds'].isin(failure_days)]['y'],
            color='darkred', s=50, label='Actual Failures (avg)')

for fdate in failure_days:
    pred_window = pd.date_range(end=fdate - pd.Timedelta(days=1), periods=lead_days)
    early_preds = agg_daily[(agg_daily['ds'].isin(pred_window)) & (agg_daily['yhat'] > failure_threshold)]
    if not early_preds.empty:
        plt.axvspan(pred_window[0], pred_window[-1], color='orange', alpha=0.3)
        earliest_pred_date = early_preds.iloc[0]['ds']
        plt.scatter(earliest_pred_date, early_preds.iloc[0]['yhat'],
                    color='orange', s=70, marker='*', label='Early Prediction (avg)')

plt.title("Overall Multi-Device Early Failure Prediction Visualization")
plt.xlabel("Date")
plt.ylabel("Failure Score (Average Across Devices)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
