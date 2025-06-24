import pandas as pd

# Read datasets
telemetry_df = pd.read_excel("telemetry.xlsx", parse_dates=["datetime"])
failures_df = pd.read_excel("machine_failures.xlsx", parse_dates=["datetime"])

# Sort by datetime for merge_asof
telemetry_df.sort_values(by=["machineID", "datetime"], inplace=True)
failures_df.sort_values(by=["machineID", "datetime"], inplace=True)

# Add a 'failure_type' column to telemetry via merge_asof
# We will merge so that each telemetry row gets the *next* failure (if any) for the same machine
# Then we filter only if failure is within the next 1 hour

consolidated = pd.merge_asof(
    telemetry_df,
    failures_df,
    on="datetime",
    by="machineID",
    direction='forward',  # get the *next* failure (if any)
    tolerance=pd.Timedelta("1H")  # only label as failure if it's within 1 hour
)

# Replace NaNs in failure column with 'none'
consolidated['failure'] = consolidated['failure'].fillna("none")

# Save to Excel
consolidated.to_excel("consolidated_machine_data.xlsx", index=False)

print("✅ Fixed consolidated dataset saved with correct failure labeling.")
