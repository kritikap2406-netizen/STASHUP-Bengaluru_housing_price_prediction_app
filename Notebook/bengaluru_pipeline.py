"""
bengaluru_pipeline.py
----------------------
Capstone Project 2: House Price Prediction System
Dataset: real Bengaluru house listings (13,320 rows)

Phases covered (per AIML Summer Internship 2026 guidelines):
  Phase 3: Data Preprocessing
  Phase 4: EDA
  Phase 5: Feature Engineering
  Phase 6: Model Building (Linear Regression, Random Forest, XGBoost)
  Phase 7: Model Evaluation (MAE, MSE, RMSE, R2)
  Phase 8 code is in ../Streamlit_App/app.py
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE = "/home/claude/BengaluruHousePrediction"
sns.set_style("whitegrid")

# %% -------------------- Phase 3: Data Preprocessing --------------------
df = pd.read_csv(f"{BASE}/Dataset/bengaluru_house_data.csv")
print("Raw shape:", df.shape)

# society has ~41% missing and is a high-cardinality free-text field
# with no predictive structure worth the noise -> drop it, keep the rest.
df = df.drop(columns=["society"])

# Drop rows with no location, size, or bath -- too little signal to impute sensibly.
df = df.dropna(subset=["location", "size", "bath"]).copy()

# Extract BHK (bedroom count) from the "size" text column, e.g. "2 BHK" / "4 Bedroom"
df["bhk"] = df["size"].apply(lambda x: int(str(x).split(" ")[0]))

# total_sqft cleanup: handles plain numbers, ranges ("2100-2850"),
# and unit suffixes (Sq. Meter, Perch, Acres, Cents, Guntha, Grounds, Sq. Yards)
UNIT_TO_SQFT = {
    "sq. meter": 10.7639,
    "sq. yards": 9.0,
    "perch": 272.25,
    "acres": 43560.0,
    "cents": 435.6,
    "guntha": 1089.0,
    "grounds": 2400.0,
}

def convert_sqft(x):
    x = str(x).strip()
    try:
        return float(x)
    except ValueError:
        pass
    if "-" in x:
        parts = x.split("-")
        try:
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
            return (lo + hi) / 2
        except ValueError:
            return None
    m = re.match(r"([\d.]+)\s*([A-Za-z. ]+)", x)
    if m:
        value, unit = float(m.group(1)), m.group(2).strip().lower()
        for key, factor in UNIT_TO_SQFT.items():
            if key in unit:
                return value * factor
    return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df[df["total_sqft"].notnull()].copy()

# balcony: fill missing with 0 (reasonable default -- many listings simply omit it)
df["balcony"] = df["balcony"].fillna(0)

# availability -> binary ready-to-move flag (most values are move-in dates)
df["is_ready_to_move"] = (df["availability"] == "Ready To Move").astype(int)
df = df.drop(columns=["availability", "size"])

print("After cleaning:", df.shape)

# %% -------------------- Phase 4: EDA --------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["price"], bins=60, kde=True, color="steelblue")
plt.xlim(0, 500)
plt.title("Univariate Analysis: Price Distribution (Lakhs INR, clipped view)")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/01_price_histogram.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df["total_sqft"].clip(upper=6000), bins=60, color="darkorange")
plt.title("Univariate Analysis: Total Sqft Distribution (clipped view)")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/02_sqft_histogram.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="area_type", y="price", hue="area_type", data=df[df["price"] < 500], palette="Set2", legend=False)
plt.title("Bivariate Analysis: Price by Area Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/03_price_by_areatype_boxplot.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
sample = df[(df["total_sqft"] < 6000) & (df["price"] < 500)]
sns.scatterplot(x="total_sqft", y="price", hue="bhk", data=sample, palette="viridis", alpha=0.5, s=25)
plt.title("Bivariate Analysis: Total Sqft vs Price")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/04_sqft_vs_price_scatter.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 6))
numeric_df = df[["total_sqft", "bath", "balcony", "bhk", "is_ready_to_move", "price"]]
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Analysis: Heatmap")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/05_correlation_heatmap.png", dpi=120)
plt.close()

print("Saved 5 EDA plots.")

# %% -------------------- Phase 5: Feature Engineering --------------------
# price_per_sqft is used ONLY as an internal signal for outlier removal below
# -- it is derived from the target, so it is dropped before modeling to avoid leakage.
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# Business-logic outliers: a home with less than ~300 sqft per bedroom is a data error
df = df[df["total_sqft"] / df["bhk"] >= 300]

# Statistical outliers: within each location, drop listings whose price/sqft
# is more than 1 std dev away from that location's mean price/sqft
def remove_price_per_sqft_outliers(data):
    out = []
    for _, sub in data.groupby("location"):
        m, s = sub["price_per_sqft"].mean(), sub["price_per_sqft"].std()
        out.append(sub[(sub["price_per_sqft"] > (m - s)) & (sub["price_per_sqft"] <= (m + s))])
    return pd.concat(out, ignore_index=True)

df = remove_price_per_sqft_outliers(df)

# Bathroom sanity check: more than (bhk + 2) bathrooms is almost always a data error
df = df[df["bath"] < df["bhk"] + 2]

# Location dimensionality reduction: locations with <= 10 listings get bucketed as "other"
df["location"] = df["location"].str.strip()
location_counts = df["location"].value_counts()
rare_locations = location_counts[location_counts <= 10].index
df["location"] = df["location"].apply(lambda x: "other" if x in rare_locations else x)

# New engineered features (beyond the basic cleaned columns)
df["bath_per_bhk"] = (df["bath"] / df["bhk"]).round(2)
df["sqft_per_bhk"] = (df["total_sqft"] / df["bhk"]).round(0)

df = df.drop(columns=["price_per_sqft"])  # drop leakage-only column

# Final outlier treatment on price itself: a handful of ultra-luxury villas/plots
# (up to Rs 2900 Lakh) sit far beyond the rest of the market and disproportionately
# skew MSE/RMSE. Cap at the 99th percentile rather than dropping, to keep sample size.
price_cap = df["price"].quantile(0.99)
df["price"] = df["price"].clip(upper=price_cap)
print(f"Capped price at 99th percentile: Rs {price_cap:.1f} Lakh")

print("After outlier removal + feature engineering:", df.shape)
print("Unique locations after reduction:", df["location"].nunique())

# %% -------------------- Phase 6: Model Building --------------------
target = "price"
categorical_features = ["location", "area_type"]
numeric_features = ["total_sqft", "bath", "balcony", "bhk", "is_ready_to_move",
                     "bath_per_bhk", "sqft_per_bhk"]

X = df[categorical_features + numeric_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42),
}

results, fitted_pipelines = {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    fitted_pipelines[name] = pipe
    print(f"\n{name}\n  MAE:{mae:.2f}  RMSE:{rmse:.2f}  R2:{r2:.4f}")

# %% -------------------- Phase 7: Model Evaluation --------------------
results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print("\n=== Model Comparison ===\n", results_df)
results_df.to_csv(f"{BASE}/Documentation/model_comparison.csv")

plt.figure(figsize=(7, 5))
sns.barplot(x=results_df.index, y=results_df["R2"], hue=results_df.index, palette="viridis", legend=False)
plt.title("Model Comparison: R2 Score")
plt.tight_layout()
plt.savefig(f"{BASE}/Documentation/plots/06_model_comparison_r2.png", dpi=120)
plt.close()

best_model_name = results_df.index[0]
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

joblib.dump(best_pipeline, f"{BASE}/Model/bengaluru_price_model.pkl")

metadata = {
    "best_model": best_model_name,
    "metrics": {k: float(v) for k, v in results[best_model_name].items()},
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "location_options": sorted(df["location"].unique().tolist()),
    "area_type_options": sorted(df["area_type"].unique().tolist()),
}
with open(f"{BASE}/Model/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model + metadata.")
