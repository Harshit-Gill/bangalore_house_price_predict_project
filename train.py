"""Train, compare and save the Bengaluru house price model.

Model selection uses 5-fold cross-validation on the TRAIN set only.
The test set is scored once, after the winner is chosen.

Outputs:
  model.pkl     - full sklearn pipeline (preprocessing + model) used by main.py
  metrics.json  - numbers for the website dashboard (copy it to the website repo)

Run: python train.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, LocalityPricePerSqft, add_features

SEED = 42
TARGET = "price_lakh"

train_df = pd.read_csv("bengaluru_train.csv")
test_df = pd.read_csv("bengaluru_test.csv")
X_train, y_train = add_features(train_df), train_df[TARGET]
X_test, y_test = add_features(test_df), test_df[TARGET]
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def linear_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("furn", OneHotEncoder(drop="first", handle_unknown="ignore"), ["furnishing_status"]),
        ("loc", Pipeline([("rate", LocalityPricePerSqft()), ("scale", StandardScaler())]),
         ["locality", "area_sqft"]),
    ])


# Unknown categories become -1, which HistGradientBoosting treats as missing.
tree_preprocessor = ColumnTransformer([
    ("num", "passthrough", NUMERIC),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
])
cat_indices = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))

candidates = {
    "linear_regression": ("Linear Regression",
                          Pipeline([("prep", linear_preprocessor()), ("model", LinearRegression())])),
    "knn": ("KNN",
            GridSearchCV(Pipeline([("prep", linear_preprocessor()), ("model", KNeighborsRegressor())]),
                         {"model__n_neighbors": [5, 10, 15, 20, 25, 40]}, cv=5,
                         scoring="neg_root_mean_squared_error")),
    "gradient_boosting": ("Gradient Boosting",
                          TransformedTargetRegressor(  # predict log(price): prices are right-skewed
                              regressor=Pipeline([
                                  ("prep", tree_preprocessor),
                                  ("model", HistGradientBoostingRegressor(
                                      categorical_features=cat_indices, max_iter=400,
                                      learning_rate=0.05, random_state=SEED)),
                              ]),
                              func=np.log1p, inverse_func=np.expm1)),
}

# 1. Compare on 5-fold CV over train only
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rmse = {}
for key, (label, est) in candidates.items():
    pred = cross_val_predict(clone(est), X_train, y_train, cv=cv)
    cv_rmse[key] = rmse(y_train, pred)
    print(f"CV RMSE  {label:<18} Rs.{cv_rmse[key]:.2f} L")

winner_key = min(cv_rmse, key=cv_rmse.get)
print(f"\nWinner by CV: {candidates[winner_key][0]}")

# 2. Fit every candidate on full train and score on test once (reported, not used for selection)
baseline_rmse = rmse(y_test, np.full(len(y_test), y_train.mean()))
fitted, test_scores = {}, {}
for key, (label, est) in candidates.items():
    model = clone(est).fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted[key] = model
    test_scores[key] = {"rmse": round(rmse(y_test, pred), 2),
                        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
                        "r2": round(float(r2_score(y_test, pred)), 3),
                        "cv_rmse": round(cv_rmse[key], 2)}
    if key == "knn":
        k = model.best_params_["model__n_neighbors"]
        candidates[key] = (f"KNN (k={k})", est)
        test_scores[key]["k"] = k
    print(f"Test     {candidates[key][0]:<18} RMSE Rs.{test_scores[key]['rmse']:.2f} L | R^2 {test_scores[key]['r2']:.3f}")

print(f"Baseline (train mean) test RMSE: Rs.{baseline_rmse:.2f} L")

winner_model = fitted[winner_key]
winner_pred = winner_model.predict(X_test)
winner_rmse = test_scores[winner_key]["rmse"]

# 3. Save the winning pipeline
joblib.dump(winner_model, "model.pkl")
print("Saved model.pkl")

# 4. Website metrics
lr = fitted["linear_regression"]
coef_names = [n.split("__")[-1] for n in lr.named_steps["prep"].get_feature_names_out()]
feature_impact = sorted(
    ({"feature": n, "coef_per_std": round(float(c), 2)} for n, c in zip(coef_names, lr.named_steps["model"].coef_)),
    key=lambda f: abs(f["coef_per_std"]), reverse=True)

errors = pd.DataFrame({"locality": test_df["locality"], "sq": (y_test - winner_pred) ** 2})
locality_rmse = (errors.groupby("locality")["sq"].agg(["size", "mean"])
                 .assign(rmse=lambda d: np.sqrt(d["mean"]))
                 .sort_values("rmse", ascending=False))

kde_x = np.linspace(y_train.min(), y_train.max(), 200)
kde_y = gaussian_kde(y_train, bw_method="scott")(kde_x)

metrics = {
    "baseline_rmse": round(baseline_rmse, 2),
    "models": test_scores,
    "winner": candidates[winner_key][0],
    "winner_key": winner_key,
    "winner_rmse": winner_rmse,
    "winner_improvement_pct": round((1 - winner_rmse / baseline_rmse) * 100, 1),
    "feature_impact": feature_impact,
    "locality_rmse": [{"locality": loc, "n_listings": int(r["size"]), "rmse": round(float(r["rmse"]), 2)}
                      for loc, r in locality_rmse.iterrows()],
    "localities": sorted(train_df["locality"].unique().tolist()),
    "furnishing_options": sorted(train_df["furnishing_status"].unique().tolist()),
    "price_distribution": {"x": np.round(kde_x, 2).tolist(), "y": np.round(kde_y, 6).tolist()},
    "train_mean_price": round(float(y_train.mean()), 2),
    "train_median_price": round(float(y_train.median()), 2),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "ranges": {
        "bhk": [int(train_df["bhk"].min()), int(train_df["bhk"].max())],
        "area_sqft": [round(float(train_df["area_sqft"].min()), 1), round(float(train_df["area_sqft"].max()), 1)],
        "bathrooms": [int(train_df["bathrooms"].min()), int(train_df["bathrooms"].max())],
        "age_years": [round(float(train_df["age_years"].min()), 1), round(float(train_df["age_years"].max()), 1)],
        "parking_spots": [int(train_df["parking_spots"].min()), int(train_df["parking_spots"].max())],
    },
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics.json (copy this into the website repo)")
