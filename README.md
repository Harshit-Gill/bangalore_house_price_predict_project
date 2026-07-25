# Bengaluru House Price API

A FastAPI backend that serves a trained K-Nearest Neighbors regression model, predicting Bengaluru house prices from listing details. Powers the live prediction tool at [harshit-gill.github.io/website](https://harshit-gill.github.io/website/).

**Live API:** https://bangalore-house-price-predict-project.onrender.com
**Docs (Swagger UI):** https://bangalore-house-price-predict-project.onrender.com/docs
**Frontend repo:** https://github.com/Harshit-Gill/website

---

## Endpoint

### `POST /predict`

**Request body:**
```json
{
  "locality": "CV Raman Nagar",
  "bhk": 3,
  "area_sqft": 2520,
  "bathrooms": 3,
  "age_years": 15,
  "parking_spots": 2,
  "furnishing_status": "Furnished"
}
```

**Response:**
```json
{
  "predicted_price_lakh": 160.07,
  "locality_recognized": true
}
```

`locality_recognized` is `false` if the locality wasn't seen during training — the model still returns a prediction, but with a fallback frequency score of 0.

### `GET /`
Health check — confirms the API is running.

---

## Model

Trained and evaluated on 8,104 real Bengaluru listings (6,483 train / 1,621 test) across 28 localities.

| Model | Test RMSE | MAE | R² |
|---|---|---|---|
| Baseline (mean price) | ₹63.31L | — | — |
| Linear Regression | ₹44.92L | ₹31.98L | 0.496 |
| **KNN (k=15) — deployed** | **₹43.06L** | **₹30.01L** | **0.537** |

KNN was chosen over Linear Regression for its lower test RMSE. It beats the baseline by 32%, exceeding the 30% improvement threshold set for this project.

**Features used:** `bhk`, `area_sqft`, `bathrooms`, `age_years`, `parking_spots`, `locality_freq` (frequency-encoded locality), `sqft_per_bhk`, plus one-hot encoded `furnishing_status`. All numeric features are standardized with `StandardScaler` before being passed to the model.

**Excluded from training** (target leakage / identifiers): `log_price`, `price_per_sqft_eda_only`, `listing_id`.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — loads artifacts, exposes `/predict` |
| `model.pkl` | Trained KNN regressor |
| `scaler.pkl` | Fitted `StandardScaler` from training data |
| `encoder.pkl` | Fitted `OneHotEncoder` for `furnishing_status` |
| `locality_freq.json` | Lookup table: locality name → frequency score |
| `requirements.txt` | Python dependencies |

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to test the endpoint interactively.

## Deployment

Deployed on Render (free tier):
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Note: free tier spins down after inactivity — the first request after idle time may take 30–50 seconds while the server wakes up.

---

Built by [Harshit Gill](https://github.com/Harshit-Gill)
