"""
CatBoost Hyperparameter Tuning

Performs time-series cross-validation to evaluate CatBoost
hyperparameter combinations and identifies the configuration
with the lowest mean absolute error for pole time prediction.

Author: Jazib Ahmed
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from itertools import product

# ----------------------------
# Load data
# ----------------------------
train = pd.read_csv("datasets/pole_mergeset.csv")
# Drop the same features from prediction
train = train.drop(columns=["Year","drs_zones_prev","active_aero_zones","Round",'type','Race','Humidity','Pressure'])

X = train.drop(columns=["PoleTime"])
y = train["PoleTime"]

# Categorical features for CatBoost
cat_features = ['tyre_wear','downforce_req', 'Rainfall','new_reg','reg']

for col in cat_features:
    X[col] = X[col].astype(str)

# ----------------------------
# Hyperparameter grid
# ----------------------------
param_grid = {
    "iterations": [1500, 2000],
    "depth": [3, 4],
    "learning_rate": [0.01, 0.025, 0.05],
}

# ----------------------------
# Time-respecting cross-validation
# ----------------------------
tscv = TimeSeriesSplit(n_splits=5)

results = []

keys = list(param_grid.keys())
combinations = list(product(*param_grid.values()))

print(f"Testing {len(combinations)} parameter combinations across {tscv.n_splits} folds...\n")

for combo in combinations:
    params = dict(zip(keys, combo))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Prediction model on current combo
        model = CatBoostRegressor(
            **params,
            cat_features=cat_features,
            random_state=42,
            verbose=False,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        # MAE so that wet qualifyings dont dominate (like they would with RMSE)
        mae = mean_absolute_error(y_val, preds)
        fold_scores.append(mae)

    avg_mae = np.mean(fold_scores)
    results.append({**params, "avg_MAE": avg_mae})
    print(f"{params} -> avg MAE: {avg_mae:.4f}")

# ----------------------------
# Show best combination
# ----------------------------
results_df = pd.DataFrame(results).sort_values("avg_MAE")
print("\nBest parameters:")
print(results_df.iloc[0])

results_df.to_csv("datasets/catboost_tuning_results.csv", index=False)