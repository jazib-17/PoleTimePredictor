"""
Future Race Dataset Builder

Generates the feature dataset for upcoming Formula 1 races by
combining historical weather, projected telemetry, and circuit
characteristics to create inputs for pole time prediction.

Author: Jazib Ahmed
"""

import fastf1
import pandas as pd
# ----------------------------
# Load completed race data
# ----------------------------
df = pd.read_csv("datasets/pole_mergeset.csv")

telemetry_features = ["TopSpeed", "AvgSpeed", "FullThrottleTime", "MostCommonGear"]

# ----------------------------
# Use the latest 4 completed seasons
# ----------------------------
recent = df[df["Year"] >= 2022]

# ----------------------------
# Average weather by circuit
# ----------------------------
weather_avg = (
    recent.groupby("Race")
    .agg({
        "AirTemp": "mean",
        "TrackTemp": "mean",
        "Humidity": "mean",
        "Pressure": "mean",
    })
    .reset_index()
)

# ----------------------------
# Average telemetry by circuit (same recent-seasons baseline as weather)
# ----------------------------
telemetry_avg = (
    recent.groupby("Race")[telemetry_features]
    .mean()
    .reset_index()
)
telemetry_avg["MostCommonGear"] = telemetry_avg["MostCommonGear"].round().astype(int)

# ----------------------------
# 2023-2025 avg -> 2026 telemetry deviation (%), per race, per feature
# ----------------------------
# 2023-2025 average
telemetry_2023_2025 = (
    df[df["Year"].isin([2023, 2024, 2025])]
    .groupby("Race")[telemetry_features]
    .mean()
    .reset_index()
    .rename(columns={f: f"{f}_2023_2025avg" for f in telemetry_features})
)
# 2026 telemetry (for races that occurred already)
telemetry_2026 = (
    df[df["Year"] == 2026][["Race"] + telemetry_features]
    .rename(columns={f: f"{f}_2026" for f in telemetry_features})
)

# Merging to aid with comparison
telemetry_compare = telemetry_2023_2025.merge(telemetry_2026, on="Race", how="inner")

# Getting percentage deviation for each telemetry feature
for feature in telemetry_features:
    telemetry_compare[f"{feature}_PctDeviation"] = (
        (telemetry_compare[f"{feature}_2026"] - telemetry_compare[f"{feature}_2023_2025avg"])
        / telemetry_compare[f"{feature}_2023_2025avg"]
    ) * 100

print("2023-2025 avg -> 2026 telemetry deviation by race:")
print(telemetry_compare[
    ["Race"] + [f"{feature}_PctDeviation" for feature in telemetry_features]
])
print(f"\nNumber of races used for this comparison: {len(telemetry_compare)}")

# Obtaining average deviation
avg_pct_deviation = {
    feature: telemetry_compare[f"{feature}_PctDeviation"].mean()
    for feature in telemetry_features
}

print("\nAverage 2023-2025 -> 2026 deviation by telemetry feature:")
for feature, pct in avg_pct_deviation.items():
    print(f"{feature}: {pct:.3f}%")

# Apply each feature's average deviation to project telemetry for 2026
for feature in telemetry_features:
    telemetry_avg[feature] = telemetry_avg[feature] * (1 + avg_pct_deviation[feature] / 100)

telemetry_avg["MostCommonGear"] = telemetry_avg["MostCommonGear"].round().astype(int)

# ----------------------------
# Remaining 2026 races
# ----------------------------

# Get the 2026 calendar
schedule = fastf1.get_event_schedule(2026)

# Get race names for rounds 7-24
race_names = (
    schedule[schedule["RoundNumber"].between(10, 24)]
    .sort_values("RoundNumber")["EventName"]
    .tolist()
)

# ----------------------------
# For 2026 only, remove the Bahrain Grand Prix
# ----------------------------
race_names.remove('Bahrain Grand Prix')

future = pd.DataFrame({
    "Year": [2026] * 13,
    "Round": list(range(10, 23)),
    "Race": race_names
})

# Merge average weather
future = future.merge(weather_avg, on="Race", how="left")

# Merge projected telemetry
future = future.merge(telemetry_avg, on="Race", how="left")

# Add rainfall (assume dry)
future["Rainfall"] = False

# ----------------------------
# For 2026 only, rename Spanish Grand Prix to Madrid Grand Prix
# ----------------------------

future["Race"] = future["Race"].replace("Spanish Grand Prix", "Madrid Grand Prix")

# Load circuit information
circuits = pd.read_csv("datasets/circuits_info.csv")

future = future.merge(circuits, on="Race", how="left")

# ----------------------------
# Arrange columns
# ----------------------------
future = future[
    [
        "Year",
        "Round",
        "Race",
        "TopSpeed",
        "AvgSpeed",
        "FullThrottleTime",
        "MostCommonGear",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "type",
        "length_km",
        "drs_zones_prev",
        "active_aero_zones",
        "tyre_wear",
        "downforce_req",
        "power_sensitivity",
        "turns"
    ]
]

# ----------------------------
# Save
# ----------------------------

future.to_csv("datasets/future_races.csv", index=False)

print("future_races.csv created successfully!")
print(future)
