# PoleTimePredictor

A machine learning pipeline that predicts Formula 1 qualifying pole times using historical race data, weather conditions, and circuit characteristics.

> **Note:** This is v1 of the project. Future versions with improvements are coming soon.

## How It Works

The project follows a three-stage pipeline:

1. **Data Extraction** (`extract_datatable.py`) — Pulls qualifying session data for every race from 2022–2026 using the [FastF1](https://github.com/theOehrly/Fast-F1) API. For each race, it records the pole-sitter's lap time along with average weather conditions during the session (air temp, track temp, humidity, pressure, rainfall).

2. **Merging & Feature Building** (`merge_tables.py`) — Combines the extracted race data with circuit-specific characteristics (track type, length, average speed, DRS zones, tyre wear, downforce requirements, etc.) to build the training set. It also constructs the feature set for upcoming 2026 races by averaging historical weather per circuit and pairing it with each track's characteristics.

3. **Prediction** (`predictor.py`) — Trains a `RandomForestRegressor` (a `HistGradientBoostingRegressor` variant is included but commented out) on the merged historical dataset, then predicts pole times for the upcoming races. It also prints feature importances to show which factors most influence pole time.

## Project Structure

| File | Description |
|---|---|
| `extract_datatable.py` | Pulls qualifying + weather data per race (2022–2026) via FastF1 → outputs `pole_dataset.csv` |
| `merge_tables.py` | Merges race data with circuit info and builds the upcoming-race feature set → outputs `pole_mergeset.csv` and `future_races.csv` |
| `predictor.py` | Trains the model and generates predictions → outputs `future_predictions.csv` |
| `circuits_info.csv` | Circuit characteristics (type, length, speed, DRS zones, tyre wear, downforce, etc.), compiled by the author |
| `pole_dataset.csv` | Raw extracted qualifying + weather data per race |
| `pole_mergeset.csv` | Final training set (race data + circuit info merged) |
| `future_races.csv` | Feature set for upcoming races to predict |
| `future_predictions.csv` | Model's predicted pole times for upcoming races |
| `all_f1_circuits.csv` | Reference circuit data (source: Kaggle, see Data Sources) |
| `f1_2026_tracks.csv` | Reference 2026 calendar/track data (source: Kaggle, see Data Sources) |

## Data Sources

- **FastF1** — live extraction of qualifying results and weather data.
- **`circuits_info.csv`** — compiled by the author.
- **[Pitwall Analytics](https://www.kaggle.com/datasets/oshomuralidaran/pitwall-analytics)** (Kaggle) — reference data used while compiling circuit info.
- **[Formula 1 Circuits 1950–Present](https://www.kaggle.com/datasets/kishan305/formula-1-circuits-1950-present?resource=download)** (Kaggle) — reference data used while compiling circuit info.

## Setup

```bash
pip install fastf1 pandas scikit-learn
```

## Usage

Run the pipeline in order:

```bash
python extract_datatable.py   # builds pole_dataset.csv
python merge_tables.py        # builds pole_mergeset.csv and future_races.csv
python predictor.py           # trains the model, outputs future_predictions.csv
```

Note: `extract_datatable.py` caches FastF1 data locally under `fastf1_cache/` to avoid repeated downloads, and can take a while to run on the first pass due to API rate limits.

## Model

- **Algorithm:** Random Forest Regressor (500 estimators)
- **Target:** Pole qualifying lap time (seconds)
- **Features:** Weather conditions (air temp, track temp, humidity, pressure, rainfall) + circuit characteristics (type, length, average speed, DRS zones, active aero zones, tyre wear, downforce requirement, power sensitivity, turns)
- Categorical features are one-hot encoded before training.

## Author

Jazib Ahmed
