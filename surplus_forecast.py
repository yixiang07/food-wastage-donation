"""
Donor Surplus Forecast
======================
Predicts daily perishable food surplus per supermarket using a
Random Forest model trained on historical sales data.

Pipeline:
    1. Load raw CSVs (sales, items, stores, transactions, oil, holidays)
    2. Filter to perishable items only
    3. Engineer target variable (7-day rolling avg − actual sales)
    4. Add temporal, macro, and store-level features
    5. Train Random Forest with 80/20 split
    6. Generate per-store surplus forecast for the most recent date
    7. Save donor_surplus_forecast.csv + visualisations

Outputs:
    donor_surplus_forecast.csv   — per-store predicted surplus with urgency labels
    eda_charts.png               — 4-panel EDA visualisation
    feature_importance.png       — ranked feature importance
    donor_dashboard.png          — donor-facing forecast dashboard

Usage:
    python surplus_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


# ── 1. Load data ─────────────────────────────────────────────
print("\n[1/10] Loading datasets...")

train = pd.read_csv("training.csv", parse_dates=["date"])
items = pd.read_csv("items.csv")
stores = pd.read_csv("stores.csv")
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
oil = pd.read_csv("oil.csv", parse_dates=["date"])
holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])

print(f"  training.csv     → {len(train):,} rows")
print(f"  items.csv        → {len(items)} rows")
print(f"  stores.csv       → {len(stores)} rows")


# ── 2. Filter to perishable items ────────────────────────────
print("\n[2/10] Filtering to perishable items...")

perishable_items = items[items["perishable"] == 1][["item_nbr", "family", "class"]]
df = train.merge(perishable_items, on="item_nbr", how="inner")

print(f"  Perishable items: {len(perishable_items)}")
print(f"  Training rows after filter: {len(df):,}")


# ── 3. Engineer target variable ──────────────────────────────
print("\n[3/10] Engineering wastage target variable...")

df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

# 7-day shifted rolling average (excludes today to avoid leakage)
df["avg_sales_7d"] = df.groupby(["store_nbr", "item_nbr"])["unit_sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

# Estimated wastage = expected − actual (floored at 0)
df["estimated_waste"] = (df["avg_sales_7d"] - df["unit_sales"]).clip(lower=0)
df = df.dropna(subset=["avg_sales_7d"])

print(f"  Avg waste per row: {df['estimated_waste'].mean():.2f} units")
print(f"  Rows with non-zero waste: {(df['estimated_waste'] > 0).sum():,}")


# ── 4. Feature engineering ───────────────────────────────────
print("\n[4/10] Engineering features...")

# Date features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Holiday flags
national_holidays = (
    holidays[(holidays["locale"] == "National") & (holidays["type"] == "Holiday")]
    [["date"]].drop_duplicates().assign(is_holiday=1)
)
df = df.drop(columns=["is_holiday"], errors="ignore")
df = df.merge(national_holidays, on="date", how="left")
df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

holiday_dates = set(national_holidays["date"])
df["is_day_after_holiday"] = df["date"].apply(
    lambda d: 1 if (d - pd.Timedelta(days=1)) in holiday_dates else 0
)

# Oil price (forward-fill weekend gaps)
oil = oil.sort_values("date")
oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
df = df.drop(columns=["dcoilwtico"], errors="ignore")
df = df.merge(oil, on="date", how="left")
df["dcoilwtico"] = df["dcoilwtico"].fillna(df["dcoilwtico"].median())

# Customer footfall
df = df.drop(columns=["transactions"], errors="ignore")
df = df.merge(transactions, on=["date", "store_nbr"], how="left")
df["transactions"] = df["transactions"].fillna(df["transactions"].median())

# Store metadata
df = df.drop(columns=["city", "type", "cluster"], errors="ignore")
df = df.merge(stores[["store_nbr", "city", "type", "cluster"]], on="store_nbr", how="left")

# Encode categoricals
le = LabelEncoder()
df["family_encoded"] = le.fit_transform(df["family"].astype(str))
df["city_encoded"] = le.fit_transform(df["city"].astype(str))
df["type_encoded"] = le.fit_transform(df["type"].astype(str))
df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

print(f"  Final shape: {df.shape}")


# ── 5. EDA charts ────────────────────────────────────────────
print("\n[5/10] Generating EDA charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Food Wastage Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Wastage by item family
waste_by_family = df.groupby("family")["estimated_waste"].mean().sort_values(ascending=False)
axes[0, 0].bar(waste_by_family.index, waste_by_family.values, color="tomato")
axes[0, 0].set_title("Average Estimated Waste by Item Family")
axes[0, 0].set_ylabel("Avg Estimated Waste (units)")
axes[0, 0].tick_params(axis="x", rotation=45)

# Holiday vs normal day
waste_by_holiday = df.groupby("is_holiday")["estimated_waste"].mean().reindex([0, 1], fill_value=0)
axes[0, 1].bar(["Normal Day", "Holiday"], waste_by_holiday.values, color=["steelblue", "orange"])
axes[0, 1].set_title("Average Wastage: Normal Day vs Holiday")
axes[0, 1].set_ylabel("Avg Estimated Waste (units)")

# Day of week
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
waste_by_dow = df.groupby("day_of_week")["estimated_waste"].mean().reindex(range(7), fill_value=0)
axes[1, 0].bar(day_names, waste_by_dow.values, color="mediumseagreen")
axes[1, 0].set_title("Average Wastage by Day of Week")
axes[1, 0].set_ylabel("Avg Estimated Waste (units)")

# Store type
waste_by_type = df.groupby("type")["estimated_waste"].mean().sort_values(ascending=False)
axes[1, 1].bar(waste_by_type.index, waste_by_type.values, color="mediumpurple")
axes[1, 1].set_title("Average Wastage by Store Type")
axes[1, 1].set_ylabel("Avg Estimated Waste (units)")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: eda_charts.png")


# ── 6. Train model ───────────────────────────────────────────
print("\n[6/10] Training Random Forest model...")

FEATURES = [
    "day_of_week", "month", "day_of_month", "is_weekend",
    "is_holiday", "is_day_after_holiday",
    "onpromotion", "dcoilwtico", "transactions",
    "type_encoded", "cluster", "city_encoded",
    "family_encoded", "avg_sales_7d",
]

X = df[FEATURES]
y = df["estimated_waste"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  MAE:  {mae:.2f} units")
print(f"  RMSE: {rmse:.2f} units")


# ── 7. Feature importance ────────────────────────────────────
print("\n[7/10] Analysing feature importance...")

importance_df = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
plt.xlabel("Importance Score")
plt.title("Feature Importance — What Drives Food Wastage?")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Saved: feature_importance.png")
for _, row in importance_df.head(3).iterrows():
    print(f"  Top driver: {row['feature']} ({row['importance']:.3f})")


# ── 8. Generate store-level forecast ─────────────────────────
print("\n[8/10] Generating donor surplus forecast...")

latest_date = df["date"].max()
today_df = df[df["date"] == latest_date].copy()

if len(today_df) == 0:
    recent = df[df["date"] >= latest_date - pd.Timedelta(days=7)]
    today_df = recent.groupby(["store_nbr", "family"]).last().reset_index()

today_df["predicted_waste"] = model.predict(today_df[FEATURES])
today_df["predicted_waste"] = today_df["predicted_waste"].clip(lower=0).round(1)

donor_view = (
    today_df.groupby("store_nbr")
    .agg(
        total_predicted_waste=("predicted_waste", "sum"),
        top_waste_category=(
            "family",
            lambda x: today_df.loc[x.index].groupby("family")["predicted_waste"].sum().idxmax(),
        ),
    )
    .reset_index()
)

donor_view = donor_view.merge(
    stores[["store_nbr", "city", "state", "type"]], on="store_nbr", how="left"
)
donor_view = donor_view.sort_values("total_predicted_waste", ascending=False)


def classify_surplus(waste):
    if waste >= 30:
        return "HIGH — reduce order or arrange pickup"
    elif waste >= 10:
        return "MODERATE — review ordering quantities"
    return "LOW — ordering well-calibrated"


donor_view["surplus_status"] = donor_view["total_predicted_waste"].apply(classify_surplus)
donor_view["forecast_date"] = latest_date.date()

donor_view = donor_view[
    ["forecast_date", "store_nbr", "city", "state", "type",
     "total_predicted_waste", "top_waste_category", "surplus_status"]
]

donor_view.to_csv("donor_surplus_forecast.csv", index=False)
print(f"  Saved: donor_surplus_forecast.csv ({len(donor_view)} stores)")


# ── 9. Donor dashboard visualisation ─────────────────────────
print("\n[9/10] Plotting donor dashboard...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"Donor Surplus Forecast Dashboard — {latest_date.date()}", fontsize=15, fontweight="bold")

top_stores = donor_view.head(15)
colors = [
    "tomato" if "HIGH" in s else "orange" if "MODERATE" in s else "mediumseagreen"
    for s in top_stores["surplus_status"]
]
axes[0].barh(
    top_stores["city"] + " (Store " + top_stores["store_nbr"].astype(str) + ")",
    top_stores["total_predicted_waste"],
    color=colors,
)
axes[0].set_xlabel("Predicted Surplus (units)")
axes[0].set_title("Top Stores by Predicted Surplus (Donor View)")
axes[0].invert_yaxis()
axes[0].legend(
    handles=[
        Patch(facecolor="tomato", label="HIGH — reduce order / arrange pickup"),
        Patch(facecolor="orange", label="MODERATE — review ordering quantities"),
        Patch(facecolor="mediumseagreen", label="LOW — ordering is well-calibrated"),
    ],
    loc="lower right", fontsize=8,
)

waste_by_family_today = (
    today_df.groupby("family")["predicted_waste"].sum().sort_values(ascending=False).head(10)
)
axes[1].bar(waste_by_family_today.index, waste_by_family_today.values, color="steelblue")
axes[1].set_title("Predicted Surplus by Item Family (All Stores)")
axes[1].set_xlabel("Item Family")
axes[1].set_ylabel("Total Predicted Surplus (units)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("donor_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: donor_dashboard.png")


# ── 10. Done ─────────────────────────────────────────────────
print("\n[10/10] Pipeline complete.")
print(f"  donor_surplus_forecast.csv is ready for the donor-facing dashboard.")
