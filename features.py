"""Feature engineering shared by train.py and main.py, so training and serving can't drift apart."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_COLUMNS = ["locality", "bhk", "area_sqft", "bathrooms", "age_years",
               "parking_spots", "furnishing_status"]
NUMERIC = ["bhk", "area_sqft", "bathrooms", "age_years", "parking_spots", "sqft_per_bhk"]
CATEGORICAL = ["furnishing_status", "locality"]
FEATURES = NUMERIC + CATEGORICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model's input columns from raw listing columns."""
    out = df[RAW_COLUMNS].copy()
    out["sqft_per_bhk"] = out["area_sqft"] / out["bhk"]
    return out[FEATURES]


class LocalityPricePerSqft(BaseEstimator, TransformerMixin):
    """Encode locality as its smoothed mean price per sqft, learned from training rows only.

    Also outputs area * that rate, an 'expected price' feature a linear model can use.
    The old locality_freq (share of listings) said how *common* an area is, not how
    *expensive*, which is why the first model priced Whitefield above Koramangala.
    """

    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, X, y):
        rate = np.asarray(y) / X["area_sqft"].to_numpy()
        stats = pd.DataFrame({"loc": X["locality"].to_numpy(), "rate": rate}).groupby("loc")["rate"].agg(["mean", "count"])
        self.global_rate_ = rate.mean()
        self.rate_map_ = ((stats["mean"] * stats["count"] + self.global_rate_ * self.smoothing)
                          / (stats["count"] + self.smoothing)).to_dict()
        return self

    def transform(self, X):
        rate = X["locality"].map(self.rate_map_).fillna(self.global_rate_).to_numpy()
        return np.c_[rate, rate * X["area_sqft"].to_numpy()]

    def get_feature_names_out(self, input_features=None):
        return np.array(["locality_price_per_sqft", "expected_price"])
