import fastf1
import pandas as pd

'''
race_df = pd.read_csv("pole_dataset.csv")
circuit_df = pd.read_csv("circuits_info.csv")

df = race_df.merge(circuit_df, on="Race", how="left")

df.to_csv("pole_mergeset.csv", index=False)
'''

# ----------------------------
# Load completed race data
# ----------------------------
df = pd.read_csv("pole_mergeset.csv")

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
        "WindSpeed": "mean",
    })
    .reset_index()
)

# ----------------------------
# Remaining 2026 races
# ----------------------------

# Get the 2026 calendar
schedule = fastf1.get_event_schedule(2026)

# Get race names for rounds 9-24
race_names = (
    schedule[schedule["RoundNumber"].between(9, 24)]
    .sort_values("RoundNumber")["EventName"]
    .tolist()
)
print(race_names)

future = pd.DataFrame({
    "Year": [2026] * 14,
    "Round": list(range(9, 23)),
    "Race": race_names
})

# ----------------------------
# Merge average weather
# ----------------------------
future = future.merge(weather_avg, on="Race", how="left")

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
# Arrange columns
# ----------------------------
future = future[
    [
        "Year",
        "Round",
        "Race",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "Rainfall",
        "type",
        "length_km",
        "avg_speed_kmh",
        "drs_zones_prev",
        "active_aero_zones",
        "tyre_wear",
        "downforce_req",
        "power_sensitivity",
        "turns",
    ]
]

# ----------------------------
# Save
# ----------------------------
future.to_csv("future_races.csv", index=False)

print("future_races.csv created successfully!")
print(future)
