"""
Pole Time Prediction

Trains a machine learning model on historical qualifying data
and predicts Formula 1 pole times for future races, while
reporting feature importance to interpret the model.

Author: Jazib Ahmed
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor


# ----------------------------
# Load datasets
# ----------------------------
train = pd.read_csv("datasets/pole_mergeset.csv")
future = pd.read_csv("datasets/future_races.csv")

# Remove the column from both datasets
train = train.drop(columns=["Year","drs_zones_prev","active_aero_zones","Round",'type','Humidity','Pressure','Race'])
future = future.drop(columns=["drs_zones_prev","active_aero_zones",'type','Humidity','Pressure'])

# ----------------------------
# Prepare training data
# ----------------------------
X_train = train.drop(columns=["PoleTime"])
y_train = train["PoleTime"]

# ----------------------------
# One-hot encode categorical columns
# ----------------------------

# For RandomForest/HistGradientBoost
'''
X_train = pd.get_dummies(X_train)

# Create encoded version for prediction

future_encoded = pd.get_dummies(future)
'''
# For CatBoost
future_encoded = future.copy()
future_encoded = future_encoded.reindex(
    columns=X_train.columns,
    fill_value=0
)

# ----------------------------
# Train Random Forest / HistGradientBoost
# ----------------------------
'''
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
'''

'''
model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.02,
    max_leaf_nodes=31,
    random_state=42
)

model.fit(X_train, y_train)
'''
# ----------------------------
# Train CatBoost
# ----------------------------

# Categorical features for CatBoost
cat_features = ['tyre_wear','downforce_req', 'Rainfall','new_reg','reg']

model = CatBoostRegressor(
    iterations=2000,
    depth=3,
    learning_rate=0.025,
    cat_features=cat_features,
    random_state=42,
    verbose=False,
)

model.fit(X_train, y_train)

# ----------------------------
# Predict future pole times
# ----------------------------
predictions = model.predict(future_encoded)

future["PredictedPoleTime"] = predictions.round(3)

# Print predictions
print(predictions.round(3))

# Save
future.to_csv("datasets/future_predictions.csv", index=False)

# Feature importances for CatBoost
importances = model.get_feature_importance()

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# Print feature importance
for _, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Save feature importance
importance_df.to_csv(
    "datasets/feature_importance.csv",
    index=False
)

# Feature importances for RandomForest
'''
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.head(30))
'''