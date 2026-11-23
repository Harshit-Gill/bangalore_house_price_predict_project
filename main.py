import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import add_features

app = FastAPI(title="Bengaluru House Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://harshit-gill.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One pipeline holds the preprocessing and the model, so it matches training exactly.
model = joblib.load("model.pkl")
with open("metrics.json") as f:
    metrics = json.load(f)
KNOWN_LOCALITIES = set(metrics["localities"])


class HouseInput(BaseModel):
    locality: str
    bhk: int = Field(ge=1, le=5)
    area_sqft: float = Field(gt=0)
    bathrooms: int = Field(ge=1, le=5)
    age_years: float = Field(ge=0)
    parking_spots: int = Field(ge=0, le=3)
    furnishing_status: Literal["Furnished", "Semi-Furnished", "Unfurnished"]


@app.get("/")
def health_check():
    return {"status": "ok", "model": metrics["winner"]}


@app.post("/predict")
def predict_price(data: HouseInput):
    features = add_features(pd.DataFrame([data.model_dump()]))
    prediction = model.predict(features)[0]
    return {
        "predicted_price_lakh": round(float(prediction), 2),
        "locality_recognized": data.locality in KNOWN_LOCALITIES,
    }
