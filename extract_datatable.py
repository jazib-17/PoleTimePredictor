"""
F1 Pole Time Dataset Builder

Creates a dataset with one row per qualifying session for use in
predicting Formula 1 pole times.

Author: Jazib Ahmed
"""

import os
import fastf1
import pandas as pd

print(fastf1.__version__)

# -----------------------------
# SETTINGS
# -----------------------------

START_YEAR = 2019
END_YEAR = 2019

OUTPUT_FILE = "pole_dataset.csv"

# -----------------------------
# LOAD EXISTING DATA (IF ANY)

cache_dir = 'fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
# -----------------------------

if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    # Set of (Year, Race) pairs we've already collected, so we don't redo them
    existing_keys = set(zip(existing_df["Year"], existing_df["Race"]))
    print(f"Found existing dataset with {len(existing_df)} rows. Will only add new races.")
else:
    existing_df = pd.DataFrame()
    existing_keys = set()
    print("No existing dataset found. Starting fresh.")

# -----------------------------
# STORAGE (only new rows go here)
# -----------------------------

dataset = []

# -----------------------------
# LOOP THROUGH YEARS
# -----------------------------

for year in range(START_YEAR, END_YEAR + 1):

    print(f"\n========== {year} ==========")

    calendar = fastf1.get_event_schedule(
        year,
        include_testing=False
    )

    races = list(calendar["EventName"])

    # -----------------------------
    # LOOP THROUGH RACES
    # -----------------------------

    for round_num, race in enumerate(races, start=1):

        # Skip races we've already processed in a previous run
        if (year, race) in existing_keys:
            print(f"Skipping {year} {race} (already in dataset)")
            continue

        print(f"Loading {year} {race} Qualifying...")

        try:
            session = fastf1.get_session(year, race, "Q")
            session.load(telemetry=True, weather=True, messages=False)
            laps = session.laps

        except fastf1.exceptions.RateLimitExceededError:
            print("Rate limit hit. Saving and stopping.")
            break

        except Exception as e:
            print(f"Skipped: {e}")
            continue

        # -----------------------------
        # Pole information
        # -----------------------------

        pole = session.results.iloc[0]

        pole_driver = pole["Abbreviation"]
        pole_team = pole["TeamName"]

        pole_lap = (
            laps
            .pick_drivers(pole_driver)
            .pick_fastest()
        )

        pole_time = pole_lap["LapTime"].total_seconds()

        # -----------------------------
        # Telemetry-derived features (pole lap only)
        # -----------------------------

        car_data = pole_lap.get_car_data().add_distance()

        top_speed = car_data["Speed"].max()
        avg_speed = car_data["Speed"].mean()

        # Most common gear (mode) - car_data["nGear"] samples throughout the lap
        most_common_gear = car_data["nGear"].mode().iloc[0]

        # Time at full throttle - sum the time deltas where Throttle == 100
        car_data["TimeDelta"] = car_data["Time"].diff().dt.total_seconds().fillna(0)
        full_throttle_time = car_data.loc[car_data["Throttle"] >= 99, "TimeDelta"].sum()

        # -----------------------------
        # Weather
        # -----------------------------

        weather = session.weather_data

        # -----------------------------
        # Store row
        # -----------------------------

        dataset.append({

            # Event
            "Year": year,
            "Round": round_num,
            "Race": race,

            # Pole
            "PoleTime": pole_time,

            # Telemetry (pole lap)
            "TopSpeed": top_speed,
            "AvgSpeed": avg_speed,
            "FullThrottleTime": full_throttle_time,
            "MostCommonGear": most_common_gear,

            # Weather
            "AirTemp": weather["AirTemp"].mean(),
            "TrackTemp": weather["TrackTemp"].mean(),
            "Humidity": weather["Humidity"].mean(),
            "Pressure": weather["Pressure"].mean(),
            "Rainfall": weather["Rainfall"].max()

        })

# -----------------------------
# CREATE DATAFRAME OF NEW ROWS
# -----------------------------

new_df = pd.DataFrame(dataset)

if not new_df.empty:
    # -----------------------------
    # COMBINE WITH EXISTING DATA
    # -----------------------------

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Drop any accidental duplicates on (Year, Race), keeping the newest
    combined_df = combined_df.drop_duplicates(subset=["Year", "Race"], keep="last")

    # Sort nicely
    combined_df = combined_df.sort_values(["Year", "Round"]).reset_index(drop=True)

    # -----------------------------
    # SAVE
    # -----------------------------

    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDataset updated successfully! Added {len(new_df)} new row(s).")
    print(combined_df.head())
    print(f"\nTotal races: {len(combined_df)}")

else:
    print("\nNo new races to add. Dataset unchanged.")
    if not existing_df.empty:
        print(f"Total races: {len(existing_df)}")