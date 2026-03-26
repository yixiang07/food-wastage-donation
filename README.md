# FoodBridge SG — Supermarket Surplus Forecasting & Beneficiary Matching

A data pipeline that tackles food wastage by predicting daily perishable surplus at supermarket level and matching NPO beneficiaries with the closest high-surplus stores for donation pickup.

Built for the IS215 Digital Business module at SMU.

## Problem

Supermarkets routinely over-order perishable goods (produce, bread, dairy, deli). Unsold stock is discarded at end of day while nearby food charities struggle to source donations. The gap is informational — stores don't know how much they'll waste, and NPOs don't know which stores have surplus.

## Solution

FoodBridge SG closes this gap with two pipelines:

1. **Donor Surplus Forecast** — a Random Forest model trained on historical sales data predicts tomorrow's surplus per store, so managers can adjust orders or schedule pickups proactively.
2. **Beneficiary Matching Engine** — an AI scoring system recommends the best stores for each NPO to collect from, weighted by surplus volume, distance, collection history, and food category preferences.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────┐
│               Donor Pipeline (analytics)            │
│                                                     │
│  Raw Sales → Perishable Filter → Wastage Proxy      │
│  → Feature Engineering → Random Forest → Forecast   │
│  → donor_surplus_forecast.csv                       │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│           Beneficiary Pipeline (dashboard)          │
│                                                     │
│  Actual Surplus → Store Geolocation → Haversine     │
│  → Match Scoring (surplus × distance × history)     │
│  → dashboard_payload.json                           │
└─────────────────────────────────────────────────────┘
```

## Repository Structure

```
├── notebooks/
│   ├── analytics.ipynb          # Donor surplus forecast (interactive, with EDA charts)
│   └── dashboard.ipynb          # Beneficiary matching engine (interactive)
├── surplus_forecast.py          # Standalone donor forecast script
├── beneficiary_dashboard.py     # Standalone beneficiary matching script
├── requirements.txt
└── .gitignore
```

## Key Methodology

### Target Variable Engineering
The dataset has no direct "wastage" column. We proxy it as:

```
estimated_waste = 7-day shifted rolling avg sales − actual sales today
```

The shift prevents data leakage (today's sales are excluded from "today's expected sales"), and the result is clipped to zero since negative waste has no physical meaning.

### Feature Engineering
The model uses 14 features across six signal categories:

| Category | Features | Rationale |
|---|---|---|
| Temporal | day_of_week, month, day_of_month, is_weekend | Wastage follows weekly/seasonal cycles |
| Holidays | is_holiday, is_day_after_holiday | Over-ordering before holidays causes day-after surplus |
| Macro | oil price (dcoilwtico) | Proxy for purchasing power — lower spending → more unsold stock |
| Footfall | transaction count | Fewer customers = more waste |
| Store | type, city, cluster | Different formats and locations have distinct waste profiles |
| Product | item family, on-promotion flag | Promotions boost sell-through; category determines spoilage rate |

### AI Matching Score (0–100)

| Signal | Weight | Logic |
|---|---|---|
| Surplus amount | 35% | More food available → higher score |
| Distance | 45% | Closer store → higher score (biggest operational constraint for NPOs) |
| Past pickups | 20% | Established collection relationships score higher |
| Category bonus | +10% | If the store's top surplus item matches the NPO's preferences |

Distance is calculated using the Haversine formula for accurate real-world kilometres between GPS coordinates.

### Surplus Classification (Donor Dashboard)

| Level | Threshold | Recommended Action |
|---|---|---|
| HIGH | ≥ 30 units | Reduce tomorrow's order or arrange donation pickup |
| MODERATE | 10–29 units | Review ordering quantities |
| LOW | < 10 units | Ordering is well-calibrated |

## Dataset

**Corporación Favorita Grocery Sales** (UCI-style supermarket data) with 7 source files: training, items, stores, testing, transactions, oil prices, and holidays/events.

## Setup

```bash
pip install -r requirements.txt
```

Place the CSV data files in the working directory, then:

```bash
# Generate donor surplus forecast
python surplus_forecast.py

# Generate beneficiary dashboard payload
python beneficiary_dashboard.py
```

Or open the notebooks for interactive exploration with EDA charts:

```bash
jupyter notebook notebooks/analytics.ipynb
```

## Outputs

| File | Consumer |
|---|---|
| `donor_surplus_forecast.csv` | Donor-facing dashboard — per-store surplus predictions |
| `dashboard_payload.json` | Beneficiary-facing frontend — metrics, AI matches, nearby stores |
| `eda_charts.png` | Exploratory analysis (wastage by family, day, holiday, store type) |
| `feature_importance.png` | Model interpretability — ranked feature importance |
| `donor_dashboard.png` | Donor dashboard visualisation |

## Tech Stack

- **pandas / NumPy** — data wrangling and feature engineering
- **scikit-learn** — Random Forest Regressor, label encoding, train/test split
- **matplotlib / seaborn** — EDA and dashboard visualisations
- **Haversine** — real-world distance calculations for store matching
