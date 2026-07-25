import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Bengaluru House Price API")

# Allow the GitHub Pages site (or any origin) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained artifacts once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("encoder.pkl")
with open("locality_freq.json") as f:
    locality_freq_map = json.load(f)

BASE_FEATURES = ['bhk', 'area_sqft', 'bathrooms', 'age_years',
                 'parking_spots', 'locality_freq', 'sqft_per_bhk']
DUMMY_FEATURES = [f'furnishing_{cat}' for cat in ohe.categories_[0][1:]]
FEATURES = BASE_FEATURES + DUMMY_FEATURES


class HouseInput(BaseModel):
    locality: str
    bhk: int = Field(ge=1, le=5)
    area_sqft: float = Field(gt=0)
    bathrooms: int = Field(ge=1, le=5)
    age_years: float = Field(ge=0)
    parking_spots: int = Field(ge=0, le=3)
    furnishing_status: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Bengaluru House Price API is running"}


@app.post("/predict")
def predict_price(data: HouseInput):
    locality_freq = locality_freq_map.get(data.locality, 0.0)
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

    furn_encoded = ohe.transform(pd.DataFrame(
        [[data.furnishing_status]], columns=['furnishing_status']
    ))
    furn_df = pd.DataFrame(furn_encoded, columns=DUMMY_FEATURES)

    full_features = pd.concat([num_features, furn_df], axis=1)[FEATURES]
    scaled = scaler.transform(full_features)

    prediction = model.predict(scaled)[0]
    return {
        "predicted_price_lakh": round(float(prediction), 2),
        "locality_recognized": data.locality in locality_freq_map,
    }
