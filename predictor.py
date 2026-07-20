import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor


# ----------------------------
# Load datasets
# ----------------------------
train = pd.read_csv("pole_mergeset.csv")
future = pd.read_csv("future_races.csv")

# Remove the column from both datasets
train = train.drop(columns=["Year","drs_zones_prev","active_aero_zones"])
future = future.drop(columns=["Year","drs_zones_prev","active_aero_zones"])

# ----------------------------
# Prepare training data
# ----------------------------
X_train = train.drop(columns=["PoleTime"])
y_train = train["PoleTime"]

# ----------------------------
# One-hot encode categorical columns
# ----------------------------
'''
X_train = pd.get_dummies(X_train)

# Create encoded version for prediction

future_encoded = pd.get_dummies(future)
'''
future_encoded = future.copy()
future_encoded = future_encoded.reindex(
    columns=X_train.columns,
    fill_value=0
)

# ----------------------------
# Train Random Forest
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

# List your categorical column names (e.g. circuit type)
cat_features = ['Race', 'type','tyre_wear', 'downforce_req', 'Rainfall']

model = CatBoostRegressor(
    iterations=2000,
    depth=4,
    learning_rate=0.01,
    cat_features=cat_features,
    random_state=42,
    verbose=False
)

model.fit(X_train, y_train)

# ----------------------------
# Predict future pole times
# ----------------------------
predictions = model.predict(future_encoded)

future["PredictedPoleTime"] = predictions.round(3)

future.to_csv("future_predictions.csv", index=False)

# Feature importances
importances = model.get_feature_importance()
for name, score in sorted(zip(X_train.columns, importances), key=lambda x: -x[1]):
    print(f"{name}: {score:.4f}")

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