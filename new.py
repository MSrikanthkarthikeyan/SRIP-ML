import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# SMAPE function (safe version of MAPE)
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Load your dataset
df = pd.read_csv("12lakh_predictive_maintenance_dataset.csv", parse_dates=['date'])

# Pick top failing device
top_device = df.groupby('device')['failure'].sum().sort_values(ascending=False).index[0]
df_device = df[df['device'] == top_device]

# Aggregate by date
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

# Smooth the failure counts
agg_df['failure'] = agg_df['failure'].rolling(window=3, min_periods=1).mean()

# Rename for Prophet
agg_df.rename(columns={'date': 'ds', 'failure': 'y'}, inplace=True)

# Split into train/test
train_df = agg_df[:-30]
test_df = agg_df[-30:]

# Train Prophet with extra regressors
model = Prophet()
regressors = [f'metric{i}' for i in range(1, 10)]
for reg in regressors:
    model.add_regressor(reg)

model.fit(train_df[['ds', 'y'] + regressors])

# Predict
future = agg_df[['ds'] + regressors].copy()
forecast = model.predict(future)

# Evaluation
merged = pd.merge(forecast[['ds', 'yhat']], test_df[['ds', 'y']], on='ds')

# Compute safe metrics
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
mae = mean_absolute_error(merged['y'], merged['yhat'])
smape_val = smape(merged['y'].values, merged['yhat'].values)
accuracy = 100 - smape_val

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Failure Forecast (Top Device)")
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()

# Show prediction summary
print("\n📊 Prediction Summary")
print("----------------------------------------")
print(f"Test Range       : {test_df['ds'].min().date()} to {test_df['ds'].max().date()}")
print(f"Total Test Days  : {len(test_df)}")
print(f"SMAPE            : {smape_val:.2f}%")
print(f"Accuracy         : {accuracy:.2f}%")
print(f"RMSE             : {rmse:.2f}")
print(f"MAE              : {mae:.2f}")
print("----------------------------------------")





# Full actual + predicted (all available data points, not just test)
full_df = pd.merge(
    agg_df[['ds', 'y']],              # actuals
    forecast[['ds', 'yhat']],         # predictions
    on='ds',
    how='left'
)

full_df.rename(columns={'y': 'actual', 'yhat': 'predicted'}, inplace=True)

# Save full results
full_df.to_csv("prophet_full_actual_vs_predicted.csv", index=False)
full_df.to_excel("prophet_full_actual_vs_predicted.xlsx", index=False)

print("✅ Exported full actual vs predicted values to:")
print(" - prophet_full_actual_vs_predicted.csv")
print(" - prophet_full_actual_vs_predicted.xlsx")



