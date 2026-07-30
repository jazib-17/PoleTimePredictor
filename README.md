# PoleTimePredictor

A machine learning pipeline that predicts Formula 1 qualifying pole times using historical race data, telemetry, weather conditions, and circuit characteristics.

> **Note:** This is v2 of the project. Future versions with improvements are coming soon.

## How It Works

The project follows a four-stage pipeline:

1. **Data Extraction** (`extract_datatable.py`) — Pulls qualifying session data using the [FastF1](https://github.com/theOehrly/Fast-F1) API, one season at a time to help manage API rate limits, appending new races to the existing dataset without redoing ones already collected. For each race, it records the pole-sitter's lap time, telemetry from the pole lap itself (top speed, average speed, time spent at full throttle, most common gear), and average weather conditions during the session (air temp, track temp, humidity, pressure, rainfall).

2. **Merging & Feature Building** (`merge_tables.py`) — Combines the extracted race data with circuit-specific characteristics (track type, length, DRS zones, tyre wear, downforce requirements, etc.) to build the training set. It also builds the feature set for the remaining 2026 races by pulling the current calendar from FastF1, averaging recent weather and telemetry (2022 onward) per circuit, and applying a correction to the telemetry features based on the actual residual difference seen in 2026 races so far, grouped by circuit type, accounting for the new 2026 regulations rather than assuming cars behave like prior years.

3. **Hyperparameter Tuning** (`parameter_testing.py`) — Runs a grid search over CatBoost parameters (iterations, depth, learning rate) using time-respecting cross-validation (`TimeSeriesSplit`), so each fold only validates on races chronologically after what it trained on. Results are ranked by average MAE across folds and saved to `catboost_tuning_results.csv`.

4. **Prediction** (`predictor.py`) — Trains a `CatBoostRegressor` on the merged historical dataset (using the best parameters found via tuning) and predicts pole times for the upcoming races. Categorical features (`Race`, `type`, `tyre_wear`, `downforce_req`, `Rainfall`) are passed natively rather than one-hot encoded. It also prints feature importances to show which factors most influence pole time.

## Versions

**v1**
- Initial pipeline: FastF1 extraction of qualifying + weather data (2022–2026), merged with a manually compiled `circuits_info.csv`.
- Upcoming races for prediction were defined as a fixed, manually written list.
- Model: `RandomForestRegressor` (500 estimators), trained on one-hot encoded weather + static circuit characteristics.

**v2**
- Backfilled historical data with 2019–2021 seasons, run year-by-year to stay within FastF1 API rate limits, and `extract_datatable.py` now appends new races to the existing dataset instead of rebuilding it from scratch each run.
- Added telemetry features derived from each pole lap: top speed, average speed, time spent at full throttle, and most common gear.
- Added a 2026 regulation correction: the observed difference between 2026 telemetry and historical averages is calculated per circuit type and applied to future race predictions, rather than assuming the new regulation era behaves like previous years.
- `merge_tables.py` now pulls the remaining 2026 calendar dynamically from FastF1's event schedule instead of a hardcoded race list.
- Added an engineered `lengthxturns` feature (circuit length × turn count).
- Dropped `WindSpeed` from the weather features — it added too much noise for the model to learn from usefully.
- Replaced the static `avg_speed_kmh` circuit characteristic with the telemetry-derived `AvgSpeed` — actual measured average lap speed for historical races, and the corrected/expected average speed (via the 2026 regulation correction) for upcoming races.
- Switched the model from Random Forest to **CatBoost**, which handles categorical features (`Race`, `type`, `tyre_wear`, `downforce_req`, `Rainfall`) natively instead of one-hot encoding.
- Dropped `Year`, `drs_zones_prev`, and `active_aero_zones` from the final feature set after testing showed they weren't pulling their weight.
- Added `parameter_testing.py` for CatBoost hyperparameter tuning via time-series cross-validation.

## Project Structure

| File | Description |
|---|---|
| `extract_datatable.py` | Pulls qualifying, telemetry, and weather data per race via FastF1 (run per-season to manage rate limits) → outputs/updates `pole_dataset.csv` |
| `merge_tables.py` | Merges race data with circuit info, applies the 2026 regulation correction, and builds the upcoming-race feature set → outputs `pole_mergeset.csv` and `future_races.csv` |
| `parameter_testing.py` | Grid search + time-series cross-validation for CatBoost hyperparameters → outputs `catboost_tuning_results.csv` |
| `predictor.py` | Trains the CatBoost model and generates predictions → outputs `future_predictions.csv` |
| `circuits_info.csv` | Circuit characteristics (type, length, DRS zones, tyre wear, downforce, etc.), compiled by the author |
| `pole_dataset.csv` | Raw extracted qualifying + telemetry + weather data per race |
| `pole_mergeset.csv` | Final training set (race data + circuit info merged) |
| `future_races.csv` | Feature set for upcoming races to predict |
| `future_predictions.csv` | Model's predicted pole times for upcoming races |
| `catboost_tuning_results.csv` | Results of the hyperparameter grid search, ranked by average MAE |
| `all_f1_circuits.csv` | Reference circuit data (source: Kaggle, see Data Sources) |
| `f1_2026_tracks.csv` | Reference 2026 calendar/track data (source: Kaggle, see Data Sources) |

## Data Sources

- **FastF1** — live extraction of qualifying results, telemetry, and weather data.
- **`circuits_info.csv`** — compiled by the author.
- **[Pitwall Analytics](https://www.kaggle.com/datasets/oshomuralidaran/pitwall-analytics)** (Kaggle) — reference data used while compiling circuit info.
- **[Formula 1 Circuits 1950–Present](https://www.kaggle.com/datasets/kishan305/formula-1-circuits-1950-present?resource=download)** (Kaggle) — reference data used while compiling circuit info.

## Setup

```bash
pip install fastf1 pandas scikit-learn catboost
```

## Usage

Run the pipeline in order:

```bash
python extract_datatable.py     # builds/updates pole_dataset.csv (run per season to manage API rate limits)
python merge_tables.py          # builds pole_mergeset.csv and future_races.csv
python parameter_testing.py     # (optional) tunes CatBoost hyperparameters
python predictor.py             # trains the model, outputs future_predictions.csv
```

Note: `extract_datatable.py` caches FastF1 data locally under `fastf1_cache/` to avoid repeated downloads. It's designed to be run one season (or a small range) at a time — new races are appended to `pole_dataset.csv` rather than the whole dataset being rebuilt, which helps avoid hitting FastF1's API rate limits.

## Model

- **Algorithm:** CatBoost Regressor
- **Target:** Pole qualifying lap time (seconds)
- **Features:** Telemetry from the pole lap — top speed, average speed (measured for historical races, corrected/expected for 2026 races), full-throttle time, most common gear — plus weather conditions (air temp, track temp, humidity, pressure, rainfall) and circuit characteristics (race, type, length, tyre wear, downforce requirement, power sensitivity, turns, length × turns)
- Categorical features (`Race`, `type`, `tyre_wear`, `downforce_req`, `Rainfall`) are passed natively to CatBoost rather than one-hot encoded.
- Best hyperparameters are selected via time-series cross-validation in `parameter_testing.py` before being used in the final model.

## Author

Jazib Ahmed
