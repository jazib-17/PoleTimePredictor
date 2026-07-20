import fastf1
import pandas as pd


race_df = pd.read_csv("pole_dataset.csv")
circuit_df = pd.read_csv("circuits_info.csv")

df = race_df.merge(circuit_df, on="Race", how="left")

df.to_csv("pole_mergeset.csv", index=False)
'''

# ----------------------------
# Load completed race data
# ----------------------------
df = pd.read_csv("pole_mergeset.csv")

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
# 2026 regulation correction, by circuit type
# ----------------------------
season_2026 = df[df["Year"] == 2026]
historical = df[df["Year"] < 2026]

historical_avg_by_race = historical.groupby("Race")[telemetry_features].mean()

residuals = (
    season_2026.set_index("Race")[telemetry_features]
    - historical_avg_by_race.reindex(season_2026["Race"])
    .set_axis(season_2026["Race"])
)

residuals = residuals.merge(
    df[["Race", "type"]].drop_duplicates(),
    left_index=True, right_on="Race", how="left"
)

residual_by_type = residuals.groupby("type")[telemetry_features].mean()
print("2026 telemetry residuals by circuit type:")
print(residual_by_type)

# ----------------------------
# Remaining 2026 races
# ----------------------------

# Get the 2026 calendar
schedule = fastf1.get_event_schedule(2026)

# Get race names for rounds 9-24
race_names = (
    schedule[schedule["RoundNumber"].between(7, 24)]
    .sort_values("RoundNumber")["EventName"]
    .tolist()
)
print(race_names)

future = pd.DataFrame({
    "Year": [2026] * 16,
    "Round": list(range(7, 23)),
    "Race": race_names
})

# ----------------------------
# Merge average weather
# ----------------------------
future = future.merge(weather_avg, on="Race", how="left")

# ----------------------------
# Merge average telemetry
# ----------------------------
future = future.merge(telemetry_avg, on="Race", how="left")

# ----------------------------
# Add rainfall (assume dry)
# ----------------------------
future["Rainfall"] = False

# ----------------------------
# Load circuit information
# ----------------------------
circuits = pd.read_csv("circuits_info.csv")

future = future.merge(circuits, on="Race", how="left")

# ----------------------------
# Apply 2026 regulation correction to telemetry, by circuit type
# ----------------------------
for feature in telemetry_features:
    correction = future["type"].map(residual_by_type[feature])
    future[feature] = future[feature] + correction.fillna(0)

future["MostCommonGear"] = future["MostCommonGear"].round()

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
        "turns",
        "lengthxturns"
    ]
]

# ----------------------------
# Save
# ----------------------------
future.to_csv("future_races.csv", index=False)

print("future_races.csv created successfully!")
print(future)
'''