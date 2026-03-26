"""
Beneficiary Dashboard Engine
=============================
Computes actual observed surplus from raw sales data, runs the
AI matching engine to recommend optimal stores for each NPO,
and outputs a JSON payload for the beneficiary-facing dashboard.

Pipeline:
    1. Load raw CSVs and compute actual daily surplus
    2. Geolocate stores and compute Haversine distances
    3. Score each store using a weighted match formula
       (surplus × distance × history × category preference)
    4. Compute dashboard metrics (total received, money saved, MoM change)
    5. Write dashboard_payload.json for the frontend

Outputs:
    dashboard_payload.json       — combined payload for frontend consumption
    dashboard_metrics.json       — summary metric cards
    ai_matches.json              — top AI-recommended stores
    nearby_stores.json           — all stores within radius
    top_categories.json          — food category breakdown
    recently_received.json       — recent pickup history

Usage:
    python beneficiary_dashboard.py
"""

import json
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────
MAX_DISTANCE_KM = 10.0       # Radius for store matching
TOP_N_MATCHES = 5            # Max match cards shown
FOOD_PRICE_PER_KG = 4.50     # SGD per kg for money-saved metric

WEIGHT_SURPLUS = 0.35
WEIGHT_DISTANCE = 0.45
WEIGHT_HISTORY = 0.20

TODAY = datetime.today().strftime("%d %b %Y")

# Beneficiary profile (in production, sourced from app database)
BENEFICIARY = {
    "id": "BEN_001",
    "name": "Food From The Heart",
    "latitude": 1.3521,
    "longitude": 103.8198,
    "preferences": ["BREAD/BAKERY", "DELI", "DAIRY EGGS", "PRODUCE"],
    "joined_date": "2024-01-15",
}

# City → (lat, lon) mapping (replace with geocoding API in production)
CITY_COORDS = {
    "Quito": (1.3521, 103.8198),
    "Guayaquil": (1.3000, 103.8000),
    "Cuenca": (1.2800, 103.8300),
    "Ambato": (1.3200, 103.7900),
    "Latacunga": (1.3700, 103.8500),
    "Riobamba": (1.3400, 103.8100),
    "Ibarra": (1.4000, 103.8700),
    "Salinas": (1.4200, 103.8000),
    "Daule": (1.3100, 103.7800),
    "Santo Domingo": (1.3600, 103.8200),
    "Cayambe": (1.3900, 103.8400),
    "Manta": (1.3300, 103.7700),
}

print("=" * 60)
print("  FoodBridge SG — Beneficiary Dashboard Engine")
print(f"  Running for: {TODAY}")
print("=" * 60)


# ── Helpers ──────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Real-world distance in km between two GPS coordinates."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def compute_match_score(surplus_units, distance_km, history_count, top_category, preferences):
    """Compute 0–100 match score for a store–beneficiary pair."""
    surplus_score = min(surplus_units / 100.0, 1.0)
    distance_score = max(0, 1.0 - (distance_km / MAX_DISTANCE_KM))
    history_score = min(history_count / 10.0, 1.0)
    category_bonus = 0.10 if str(top_category).upper() in [p.upper() for p in preferences] else 0.0

    raw = (
        WEIGHT_SURPLUS * surplus_score
        + WEIGHT_DISTANCE * distance_score
        + WEIGHT_HISTORY * history_score
        + category_bonus
    )
    return round(min(raw, 1.0) * 100, 1)


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filename}")


# ── 1. Load raw data ─────────────────────────────────────────
print("\n[1/7] Loading raw data...")

train = pd.read_csv("training.csv", parse_dates=["date"])
items = pd.read_csv("items.csv")
stores = pd.read_csv("stores.csv")
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
oil = pd.read_csv("oil.csv", parse_dates=["date"])
holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])

print(f"  train: {len(train):,} rows | items: {len(items)} | stores: {len(stores)}")


# ── 2. Compute actual daily surplus ──────────────────────────
print("\n[2/7] Computing actual daily surplus...")

perishables = items[items["perishable"] == 1][["item_nbr", "family"]]
df = train.merge(perishables, on="item_nbr", how="inner")
df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

df["avg_sales_7d"] = df.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
df["estimated_waste"] = (df["avg_sales_7d"] - df["unit_sales"]).clip(lower=0)
df = df.dropna(subset=["avg_sales_7d"])

latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()

surplus = (
    latest_df.groupby("store_nbr")
    .agg(
        total_predicted_waste=("estimated_waste", "sum"),
        top_waste_category=(
            "family",
            lambda x: latest_df.loc[x.index].groupby("family")["estimated_waste"].sum().idxmax(),
        ),
    )
    .reset_index()
)
surplus = surplus.merge(
    stores[["store_nbr", "city", "state", "type"]], on="store_nbr", how="left"
)
surplus["forecast_date"] = latest_date.date()

print(f"  Actual surplus computed for {latest_date.date()} — {len(surplus)} stores")


# ── 3. Beneficiary pickup history ────────────────────────────
print("\n[3/7] Loading beneficiary profile and history...")

pickup_history = pd.DataFrame([
    {"item_name": "Vegetables", "quantity_kg": 30, "store_name": "FairPrice Xtra",
     "store_nbr": 3, "distance_km": 2.3,
     "pickup_date": (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"), "status": "pending"},
    {"item_name": "Condiments", "quantity_kg": 20, "store_name": "FairPrice Xtra",
     "store_nbr": 3, "distance_km": 2.3,
     "pickup_date": (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d"), "status": "complete"},
    {"item_name": "Rice", "quantity_kg": 40, "store_name": "FairPrice Xtra",
     "store_nbr": 3, "distance_km": 2.3,
     "pickup_date": (datetime.today() - timedelta(days=6)).strftime("%Y-%m-%d"), "status": "complete"},
    {"item_name": "Bread", "quantity_kg": 25, "store_name": "Cold Storage",
     "store_nbr": 7, "distance_km": 2.7,
     "pickup_date": (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d"), "status": "complete"},
    {"item_name": "Dairy", "quantity_kg": 15, "store_name": "NTUC FairPrice",
     "store_nbr": 11, "distance_km": 3.4,
     "pickup_date": (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d"), "status": "complete"},
    {"item_name": "Bread", "quantity_kg": 35, "store_name": "FairPrice Xtra",
     "store_nbr": 3, "distance_km": 2.3,
     "pickup_date": (datetime.today() - timedelta(days=20)).strftime("%Y-%m-%d"), "status": "complete"},
])

pickup_history["pickup_date"] = pd.to_datetime(pickup_history["pickup_date"])

history_counts = (
    pickup_history[pickup_history["status"] == "complete"]
    .groupby("store_nbr")["item_name"]
    .count()
    .reset_index()
    .rename(columns={"item_name": "collections"})
)

print(f"  Profile loaded — {len(pickup_history)} past pickups")


# ── 4. Geolocate stores ─────────────────────────────────────
print("\n[4/7] Mapping store coordinates...")

stores["latitude"] = stores["city"].apply(lambda c: CITY_COORDS.get(c, (1.3521, 103.8198))[0])
stores["longitude"] = stores["city"].apply(lambda c: CITY_COORDS.get(c, (1.3521, 103.8198))[1])


# ── 5. AI matching engine ────────────────────────────────────
print("\n[5/7] Running AI matching engine...")

cols_to_drop = [c for c in ["city", "state", "type"] if c in surplus.columns]
surplus_with_loc = surplus.drop(columns=cols_to_drop).merge(
    stores[["store_nbr", "city", "state", "type", "latitude", "longitude"]],
    on="store_nbr", how="left",
)

matches, nearby = [], []
ben_lat, ben_lon = BENEFICIARY["latitude"], BENEFICIARY["longitude"]
prefs = BENEFICIARY["preferences"]

for _, store in surplus_with_loc.iterrows():
    if pd.isna(store["latitude"]):
        continue

    dist = haversine_km(ben_lat, ben_lon, store["latitude"], store["longitude"])
    if dist > MAX_DISTANCE_KM:
        continue

    hist_row = history_counts[history_counts["store_nbr"] == store["store_nbr"]]
    hist_cnt = int(hist_row["collections"].values[0]) if len(hist_row) > 0 else 0

    score = compute_match_score(
        surplus_units=store["total_predicted_waste"],
        distance_km=dist,
        history_count=hist_cnt,
        top_category=store.get("top_waste_category", ""),
        preferences=prefs,
    )

    record = {
        "store_nbr": int(store["store_nbr"]),
        "store_name": f"{store['city']} Store #{int(store['store_nbr'])}",
        "city": store["city"],
        "store_type": store["type"],
        "distance_km": round(dist, 1),
        "predicted_waste": round(store["total_predicted_waste"], 0),
        "top_category": store.get("top_waste_category", "N/A"),
        "past_pickups": hist_cnt,
        "match_score": score,
        "surplus_status": (
            "High surplus" if store["total_predicted_waste"] >= 30
            else "Moderate surplus" if store["total_predicted_waste"] >= 10
            else "Low surplus"
        ),
        "is_matched": score >= 60,
    }
    nearby.append(record)
    if score >= 60:
        matches.append(record)

matches.sort(key=lambda x: x["match_score"], reverse=True)
nearby.sort(key=lambda x: x["predicted_waste"], reverse=True)
matches = matches[:TOP_N_MATCHES]

print(f"  {len(matches)} AI matches | {len(nearby)} nearby stores")


# ── 6. Dashboard metrics ────────────────────────────────────
print("\n[6/7] Computing dashboard metrics...")

completed = pickup_history[pickup_history["status"] == "complete"]
pending = pickup_history[pickup_history["status"] == "pending"]

total_received_kg = int(completed["quantity_kg"].sum())
money_saved = round(total_received_kg * FOOD_PRICE_PER_KG, 2)

cutoff_30 = datetime.today() - timedelta(days=30)
cutoff_60 = datetime.today() - timedelta(days=60)

recent_kg = completed[completed["pickup_date"] >= cutoff_30]["quantity_kg"].sum()
prior_kg = completed[
    (completed["pickup_date"] >= cutoff_60) & (completed["pickup_date"] < cutoff_30)
]["quantity_kg"].sum()

pct_change = round((recent_kg - prior_kg) / prior_kg * 100, 1) if prior_kg > 0 else 0

category_counts = completed.groupby("item_name")["quantity_kg"].sum().sort_values(ascending=False)
top_category = category_counts.index[0] if len(category_counts) > 0 else prefs[0]

# Top categories breakdown
max_qty = category_counts.max() if len(category_counts) > 0 else 1
top_categories = [
    {"category": cat, "quantity_kg": int(qty), "percentage": round(qty / max_qty * 100, 0)}
    for cat, qty in category_counts.head(5).items()
]

# Recently received
recently_received = []
for _, row in pickup_history.sort_values("pickup_date", ascending=False).head(5).iterrows():
    delta = (datetime.today() - pd.to_datetime(row["pickup_date"])).days
    days_label = "Today" if delta == 0 else "Yesterday" if delta == 1 else f"{delta} days ago"
    recently_received.append({
        "item_name": row["item_name"],
        "quantity_kg": int(row["quantity_kg"]),
        "store_name": row["store_name"],
        "distance_km": row["distance_km"],
        "days_ago": days_label,
        "status": row["status"],
        "pickup_date": str(row["pickup_date"])[:10],
    })

metrics = {
    "total_received_kg": total_received_kg,
    "total_received_label": f"{total_received_kg} kg",
    "pct_change_received": pct_change,
    "money_saved": money_saved,
    "money_saved_label": f"${money_saved:,.0f}",
    "pct_change_saved": pct_change,
    "ai_matches_today": len(matches),
    "pending_pickups": len(pending),
    "top_category": top_category,
    "last_updated": datetime.now().strftime("%a %d %b %Y, %I:%M %p"),
    "beneficiary_name": BENEFICIARY["name"],
}

print(f"  Total received: {metrics['total_received_label']} ({'+' if pct_change >= 0 else ''}{pct_change}% MoM)")
print(f"  Money saved:    {metrics['money_saved_label']}")
print(f"  AI matches:     {len(matches)}")


# ── 7. Write JSON outputs ───────────────────────────────────
print("\n[7/7] Writing dashboard JSON files...")

save_json(metrics, "dashboard_metrics.json")
save_json(matches, "ai_matches.json")
save_json(top_categories, "top_categories.json")
save_json(nearby, "nearby_stores.json")
save_json(recently_received, "recently_received.json")

save_json({
    "generated_at": datetime.now().isoformat(),
    "metrics": metrics,
    "ai_matches": matches,
    "top_categories": top_categories,
    "nearby_stores": nearby,
    "recently_received": recently_received,
}, "dashboard_payload.json")

print("\n" + "=" * 60)
print("  BENEFICIARY DASHBOARD ENGINE COMPLETE")
print("=" * 60)
print("  dashboard_payload.json is ready for your frontend.")
print("=" * 60)
