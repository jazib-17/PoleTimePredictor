"""
Pole Dataset Merger

Filters out qualifying sessions from outdated circuit layouts,
merges circuit characteristics with the pole time dataset,
and exports the final machine learning dataset.

Author: Jazib Ahmed
"""

import pandas as pd

race_df = pd.read_csv("datasets/pole_dataset.csv")
circuit_df = pd.read_csv("datasets/circuits_info.csv")

# ----------------------------
# Drop pre-2023 Singapore, pre-2023 Spanish, pre-2020 Australian races (track layout change)
# ----------------------------
race_df = race_df[
~(
    (
        race_df["Race"].str.contains("Singapore", case=False, na=False)
        & race_df["Year"].isin([2019, 2022])
    )
    |
    (
        race_df["Race"].str.contains("Spanish", case=False, na=False)
        & race_df["Year"].between(2019, 2022)
    )
    |
    (
        race_df["Race"].str.contains("Australian", case=False, na=False)
        & (race_df["Year"] == 2019)
    )
)
].reset_index(drop=True)

# ----------------------------
# Standardize race names
# ----------------------------
race_df["Race"] = race_df["Race"].replace({
    "Mexican Grand Prix": "Mexico City Grand Prix",
    "Brazilian Grand Prix": "São Paulo Grand Prix".
    "Barcelona Grand Prix": "Spanish Grand Prix"
})

# Merge the circuit information and pole laptime info datasets by race
df = race_df.merge(circuit_df, on="Race", how="left")

# Save
df.to_csv("datasets/pole_mergeset.csv", index=False)