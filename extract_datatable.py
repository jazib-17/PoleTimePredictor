"""
F1 Pole Time Dataset Builder

Creates a dataset with one row per qualifying session for use in
predicting Formula 1 pole times.

Author: Jazib Ahmed
"""

import fastf1
import pandas as pd

print(fastf1.__version__)

# -----------------------------
# SETTINGS
# -----------------------------

START_YEAR = 2022
END_YEAR = 2026

fastf1.Cache.enable_cache("fastf1_cache")

# -----------------------------
# STORAGE
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

        print(f"Loading {year} {race} Qualifying...")

        try:
            session = fastf1.get_session(year, race, "Q")
            session.load(telemetry=False, weather=True, messages=False)
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

            # Weather
            "AirTemp": weather["AirTemp"].mean(),
            "TrackTemp": weather["TrackTemp"].mean(),
            "Humidity": weather["Humidity"].mean(),
            "Pressure": weather["Pressure"].mean(),
            "Rainfall": weather["Rainfall"].max()

        })

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

df = pd.DataFrame(dataset)

# Sort nicely
df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

# -----------------------------
# SAVE
# -----------------------------

df.to_csv("pole_dataset.csv", index=False)

print("\nDataset created successfully!")
print(df.head())
print(f"\nTotal races: {len(df)}")