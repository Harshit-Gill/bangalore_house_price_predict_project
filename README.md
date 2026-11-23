# Bengaluru House Price API

A FastAPI backend that serves a gradient boosting regression model, predicting Bengaluru house prices from listing details. Powers the live prediction tool at [harshit-gill.github.io/website](https://harshit-gill.github.io/website/).

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

`locality_recognized` is `false` if the locality wasn't seen during training — the model still returns a prediction, treating the locality as unknown.

### `GET /`
Health check — confirms the API is running.

---

## Model

Trained and evaluated on 8,104 Bengaluru listings (6,483 train / 1,621 test) across 28 localities.
The model was chosen by 5-fold cross-validation on the training set; the test set was scored once at the end.

| Model | CV RMSE (train) | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Baseline (mean price) | — | ₹63.31L | — | — |
| Linear Regression | ₹27.70L | ₹29.18L | ₹17.99L | 0.787 |
| KNN (k=20, chosen by CV) | ₹28.14L | ₹29.53L | ₹18.25L | 0.782 |
| **Gradient Boosting, log price — deployed** | **₹26.87L** | **₹27.68L** | **₹16.20L** | **0.809** |

The deployed model beats the mean-price baseline by 56%.

### What changed in v2

Version 1 encoded locality as `locality_freq`, the share of listings in each area. That measures how *common* an area is, not how *expensive* it is, so the model priced a 2 BHK in Whitefield (₹108L) above the same flat in Koramangala (₹96L), and reached only R² 0.50.

Version 2 fixes this:
- **Gradient boosting** uses locality directly as a categorical feature and predicts log(price), since prices are right-skewed.
- **Linear regression and KNN** encode locality as its smoothed mean price per sqft, learned from training rows only.
- **KNN's k is chosen by cross-validation**, not by test score.
- **Preprocessing and model are saved as one sklearn pipeline**, so the API cannot drift from training.

Same 2 BHK, 1200 sqft flat after the fix: Koramangala ₹148L, Whitefield ₹96L, Sadashivnagar ₹163L.

**Features:** `bhk`, `area_sqft`, `bathrooms`, `age_years`, `parking_spots`, `sqft_per_bhk`, `locality`, `furnishing_status`.

**Excluded** (target leakage / identifiers): `log_price`, `price_per_sqft_eda_only`, `listing_id`.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: loads the pipeline, exposes `/predict` |
| `features.py` | Feature engineering shared by training and the API |
| `train.py` | Compares models with CV, saves `model.pkl` and `metrics.json` |
| `model.pkl` | Trained pipeline (preprocessing + gradient boosting) |
| `metrics.json` | Evaluation numbers used by the website dashboard |
| `data_cleaning.py` | Data audit and cleaning that produced the train/test CSVs |
| `main2.py` | Original v1 training script, kept for reference |
| `requirements.txt` | Pinned Python dependencies (scikit-learn must match the version that saved `model.pkl`) |

To retrain: `python train.py`

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
