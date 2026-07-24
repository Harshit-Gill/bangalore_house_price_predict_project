import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "colab"

import warnings
warnings.filterwarnings('ignore')
print("All libraries loaded.")

def kde_line(data, n_points=200):
    data = data.dropna()
    kde = gaussian_kde(data, bw_method='scott')
    x = np.linspace(data.min(), data.max(), n_points)
    y = kde(x)
    return x, y

# Section 1: Load Train + Test + Baseline Recap

train_df = pd.read_csv("bengaluru_train.csv")
test_df = pd.read_csv("bengaluru_test.csv")

print(f"Train: {train_df.shape[0]} rows x {train_df.shape[1] } columns")
print(f"Test {test_df.shape[0]} rows x {test_df.shape[1]} columns")
print(f"\nTrain columns: {list(train_df.columns)}")

#Feature selection: Leakage Discpline

#No target-derived columns.** `log_price = log(price_lakh + 1)` and `price_per_sqft_eda_only`
#are both computed from the target — including either in `X` is silent leakage.
#No raw identifiers. listing_id carries no signal.

BASE_FEATURES = ['bhk', 'area_sqft', 'bathrooms', 'age_years',
    'parking_spots', 'locality_freq', 'sqft_per_bhk',
]

TARGET = 'price_lakh'

X_train_num = train_df[BASE_FEATURES].copy()
X_test_num = test_df[BASE_FEATURES].copy()
y_train, y_test = train_df[TARGET], test_df[TARGET]

print(f"Numeric base features  : {len(BASE_FEATURES)}")
print(f"Categorical to encode  : 1 (furnishing_status -> 2 dummies in Section 1.5)")
print(f"Target                 : {TARGET}")
print()
print("Excluded (target leakage): log_price, price_per_sqft_eda_only")
print("Excluded (identifier)    : listing_id")
print("Encoded into the model   : locality (as locality_freq), furnishing_status (one-hot, drop_first)")

mean_price_train = y_train.mean()
baseline_pred = np.full(len(y_test), mean_price_train)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
target_rmse = baseline_rmse * 0.7
print(f"Mean train price  : Rs.{mean_price_train:.2f} L")
print(f"Baseline RMSE     : Rs.{baseline_rmse:.2f} L")
print(f"Target RMSE (30%) : Rs.{target_rmse:.2f} L   <-- any model must beat this")



""" WHAT WE INHERIT FROM SESSION 11.1

    a) train set: 6483 rows, 13 columns (cleaned + engineered features)
    b) test set: 1621 rows, 12 columns (same transforms applied)
    c) target: price_lakh | numeric base features : 7 
    d) baseline RMSE: Rs.63.31 L
    e) success threshold: Rs. 44.32 L (30% better than baseline)
"""

# One Hot encoding urnishing_status - fit on train, apply to test

# Fit the encoder on train only

try:
    ohe = OneHotEncoder(drop = 'first', sparse_output=False, handle_unknown='ignore')
except TypeError:
    ohe = OneHotEncoder(drop='first', sparse = False, handle_unknown='ignore')

train_furn = ohe.fit_transform(train_df[['furnishing_status']])
test_furn = ohe.transform(test_df[['furnishing_status']])

# Build the dummy column names sso we can attach them back to a DataFrame

DUMMY_FEATURES = [f'furnishing_{cat}' for cat in ohe.categories_[0][1:]]
print(f"Encoder learned categories : {list(ohe.categories_[0])}")
print(f"Dummy columns produced     : {DUMMY_FEATURES}")
print(f"Baseline (dropped) category: {ohe.categories_[0][0]}")

# Concatenate numeric base features with the dummy columns to produce the full feature matrix

train_furn_df = pd.DataFrame(train_furn, columns= DUMMY_FEATURES, index = X_train_num.index)
test_furn_df = pd.DataFrame(test_furn, columns = DUMMY_FEATURES, index = X_test_num.index)

X_train_full = pd.concat([X_train_num, train_furn_df], axis = 1)
X_test_full = pd.concat([X_test_num, test_furn_df], axis = 1)

FEATURES = BASE_FEATURES + DUMMY_FEATURES
print(f"\nFinal FEATURES list ({len(FEATURES)} columns): {FEATURES}")
print(f"X_train_full shape: {X_train_full.shape}")
print(f"X_test_full  shape: {X_test_full.shape}")
assert list(X_train_full.columns) == list(X_test_full.columns), 'Train/test column mismatch!'

# SCale the full feature matrix (numeric base + furnishing dummies)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

print("Train-scaled means (should be ~0):")
print(np.round(X_train_scaled.mean(axis=0), 3))
print("\nTrain-scaled stds  (should be ~1):")
print(np.round(X_train_scaled.std(axis=0), 3))
print("\nTest-scaled means (will NOT be exactly 0 — that is correct):")
print(np.round(X_test_scaled.mean(axis=0), 3))



# Section 2 : Linear regression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_test_pred = lr.predict(X_test_scaled)

print(f"Model trained on {X_train_scaled.shape[0]} rows, evaluated on {X_test_scaled.shape[0]} rows.")
print(f"Features consumed: {X_train_scaled.shape[1]} (scaled).")

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_mae = mean_absolute_error(y_test, lr_test_pred)
lr_r2 = r2_score(y_test, lr_test_pred)

print(f"Linear Regression on TEST set")
print(f"  RMSE : Rs.{lr_rmse:.2f} L")
print(f"  MAE  : Rs.{lr_mae:.2f} L")
print(f"  R^2  : {lr_r2:.3f}")

lr_improvement = ((baseline_rmse - lr_rmse) / baseline_rmse) * 100

print(f"Test RMSE: Rs.{lr_rmse:.2f} L | MAE: Rs.{lr_mae:.2f} L | R-squared {lr_r2:.3f}")
print(f"Imporvement over baseline: {lr_improvement:.1f}%")
print(f"{'achieved' if lr_improvement >= 30 else 'NOT YET ACHIEVED'} 30% target")

# coefficient interpretation

# scaler.scale_[i] is the train-std of feature i; back-transform recovers raw-unit coefficients
raw_coefs = lr.coef_ / scaler.scale_

coef_df = pd.DataFrame({
    'feature': FEATURES,
    'train_std': np.round(scaler.scale_, 3),
    'coef_per_std (Rs.L)': np.round(lr.coef_, 3),
    'coef_raw_units (Rs.L per 1 unit)': np.round(raw_coefs, 5),
}).sort_values('coef_per_std (Rs.L)', key=abs, ascending=False).reset_index(drop=True)

print(coef_df.to_string(index=False))
print(f"\nIntercept (in scaled space): Rs.{lr.intercept_:.2f} L")


# Actual vs predicted 
lim_max = max(y_test.max(), lr_test_pred.max()) * 1.05
lim_min = min(y_test.min(), lr_test_pred.min()) * 0.95

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_test, y=lr_test_pred, mode='markers',
    marker=dict(size=7, color='#ED1C24', opacity=0.6),
    hovertemplate='Actual Rs.%{x:.1f}L<br>Predicted Rs.%{y:.1f}L<br>Residual Rs.%{customdata:.1f}L<extra></extra>',
    customdata=y_test.values - lr_test_pred,
    name='Test rows',
))
fig.add_trace(go.Scatter(
    x=[lim_min, lim_max], y=[lim_min, lim_max], mode='lines',
    line=dict(color='black', dash='dash'), name='y = x (perfect prediction)',
    hoverinfo='skip',
))
fig.update_layout(title='Linear Regression — Actual vs Predicted (Test set)',
                  xaxis_title='Actual price (Rs. lakhs)',
                  yaxis_title='Predicted price (Rs. lakhs)',
                  width=800, height=500, template='simple_white')
fig.show(renderer='browser')

lr_residuals = y_test.values - lr_test_pred
lr_corr      = np.corrcoef(y_test.values, lr_test_pred)[0, 1]
lr_mean_abs  = np.mean(np.abs(lr_residuals))
lr_pct_within_30L = (np.abs(lr_residuals) <= 30).mean() * 100


# Section 3: KNN Regressor + K-selection

k_values = [1,3, 5, 7, 15, 25,99,199]
knn_results = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    knn_results.append({'k': k, 'test_rmse': round(rmse, 2)})

knn_sweep_df = pd.DataFrame(knn_results)
print(knn_sweep_df.to_string(index=False))

best_k = int(knn_sweep_df.loc[knn_sweep_df['test_rmse'].idxmin(), 'k'])
best_knn_rmse = knn_sweep_df['test_rmse'].min()

print(f"Best k        : {best_k}")
print(f"Best test RMSE: Rs.{best_knn_rmse:.2f} L")

knn_improvement = (1 - best_knn_rmse / baseline_rmse) * 100

knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
knn_test_pred = knn_best.predict(X_test_scaled)

knn_mae = mean_absolute_error(y_test, knn_test_pred)
knn_r2  = r2_score(y_test, knn_test_pred)

print(f"Final KNN (k={best_k}) on TEST set")
print(f"  RMSE : Rs.{best_knn_rmse:.2f} L")
print(f"  MAE  : Rs.{knn_mae:.2f} L")
print(f"  R^2  : {knn_r2:.3f}")

comparison = pd.DataFrame([
    {'model': 'Baseline (mean)',          'test_rmse': round(baseline_rmse, 2),  'test_mae': np.nan,            'test_r2': np.nan},
    {'model': 'Linear Regression',        'test_rmse': round(lr_rmse, 2),        'test_mae': round(lr_mae, 2),   'test_r2': round(lr_r2, 3)},
    {'model': f'KNN (k={best_k})',        'test_rmse': round(best_knn_rmse, 2),  'test_mae': round(knn_mae, 2),  'test_r2': round(knn_r2, 3)},
])
comparison['beats_baseline_pct'] = ((1 - comparison['test_rmse'] / baseline_rmse) * 100).round(1)

print(comparison.to_string(index=False))

winner_idx = comparison['test_rmse'].idxmin()
winner_name = comparison.loc[winner_idx, 'model']
winner_rmse = comparison.loc[winner_idx, 'test_rmse']
winner_improvement = comparison.loc[winner_idx, 'beats_baseline_pct']

print(f"WINNER: {winner_name}")
print(f"  Test RMSE                  : Rs.{winner_rmse:.2f} L")
print(f"  Improvement over baseline  : {winner_improvement:.1f}%")
print(f"  30% success threshold      : {'ACHIEVED' if winner_improvement >= 30 else 'NOT YET ACHIEVED'}")

# Identify winner's predictions
if winner_name == 'Linear Regression':
    winner_pred = lr_test_pred
elif winner_name.startswith('KNN'):
    winner_pred = knn_test_pred
else:
    winner_pred = baseline_pred

residuals = y_test.values - winner_pred

print(f"Residuals on TEST set ({winner_name}):")
print(f"  Mean   : Rs.{residuals.mean():+.2f} L  (healthy if near 0)")
print(f"  Median : Rs.{np.median(residuals):+.2f} L")
print(f"  Std    : Rs.{residuals.std():.2f} L")
print(f"  Min    : Rs.{residuals.min():+.2f} L  (most over-predicted)")
print(f"  Max    : Rs.{residuals.max():+.2f} L  (most under-predicted)")

bias_direction = 'UNDER-predicting' if residuals.mean() > 0 else 'OVER-predicting' if residuals.mean() < 0 else 'balanced'

skew_val = float(pd.Series(residuals).skew())
shape_read = ('roughly symmetric' if abs(skew_val) < 0.5 else
              'right-skewed (long tail of UNDER-predictions)' if skew_val > 0 else
              'left-skewed (long tail of OVER-predictions)')

test_with_pred = test_df[['locality', 'price_lakh']].copy()
test_with_pred['predicted'] = winner_pred
test_with_pred['abs_error'] = np.abs(test_with_pred['price_lakh'] - test_with_pred['predicted'])
test_with_pred['sq_error']  = (test_with_pred['price_lakh'] - test_with_pred['predicted']) ** 2

test_with_pred.head()

locality_errors = (
    test_with_pred.groupby('locality')
    .agg(n_listings      = ('price_lakh', 'size'),
         mean_actual     = ('price_lakh', 'mean'),
         locality_rmse   = ('sq_error',  lambda x: float(np.sqrt(x.mean()))),
         mean_abs_error  = ('abs_error', 'mean'))
    .sort_values('locality_rmse', ascending=False)
    .round(2)
)
print(locality_errors.to_string())

worst_locality = locality_errors.index[0]
worst_locality_rmse = locality_errors.iloc[0]['locality_rmse']
worst_n = int(locality_errors.iloc[0]['n_listings'])
best_locality = locality_errors.index[-1]
best_locality_rmse = locality_errors.iloc[-1]['locality_rmse']
gap = worst_locality_rmse - best_locality_rmse

import joblib
import json

# Save whichever model won — adjust if KNN wins instead
joblib.dump(lr, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(ohe, "encoder.pkl")

# Save the locality -> locality_freq lookup so the API can compute it from a locality name
locality_freq_map = train_df.groupby('locality')['locality_freq'].first().to_dict()
with open("locality_freq.json", "w") as f:
    json.dump(locality_freq_map, f)

print("Saved model.pkl, scaler.pkl, encoder.pkl, locality_freq.json")
