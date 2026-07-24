import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load saved artifacts once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("encoder.pkl")
with open("locality_freq.json") as f:
    locality_freq_map = json.load(f)

BASE_FEATURES = ['bhk', 'area_sqft', 'bathrooms', 'age_years',
                 'parking_spots', 'locality_freq', 'sqft_per_bhk']

class HouseInput(BaseModel):
    bhk: int
    area_sqft: float
    bathrooms: int
    age_years: float
    parking_spots: int
    locality: str
    furnishing_status: str

@app.post("/predict")
def predict_price(data: HouseInput):
    locality_freq = locality_freq_map.get(data.locality, 0)  # 0 for unseen locality
    sqft_per_bhk = data.area_sqft / data.bhk

    num_features = pd.DataFrame([{
        'bhk': data.bhk,
        'area_sqft': data.area_sqft,
        'bathrooms': data.bathrooms,
        'age_years': data.age_years,
        'parking_spots': data.parking_spots,
        'locality_freq': locality_freq,
        'sqft_per_bhk': sqft_per_bhk,
    }])

    furn_encoded = ohe.transform(pd.DataFrame([[data.furnishing_status]], columns=['furnishing_status']))
    dummy_cols = [f'furnishing_{cat}' for cat in ohe.categories_[0][1:]]
    furn_df = pd.DataFrame(furn_encoded, columns=dummy_cols)

    full_features = pd.concat([num_features, furn_df], axis=1)
    scaled = scaler.transform(full_features)

    prediction = model.predict(scaled)[0]
    return {"predicted_price_lakh": round(float(prediction), 2)}