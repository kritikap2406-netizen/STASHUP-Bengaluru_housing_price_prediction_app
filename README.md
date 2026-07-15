# 🏡 Bengaluru House Price Prediction

An ML-powered web app that estimates residential house prices in Bengaluru based on location, area, BHK, bathrooms, and more — built as part of the **AIML Summer Internship 2026 Capstone Project (IIHMF, MNNIT Allahabad)**.

🔗 **[Try the Live App →](https://kritikap2406-netizen-stashup-bengaluru--streamlit-appapp-k80y5b.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-016A70)
![License](https://img.shields.io/badge/status-active-brightgreen)

---

## 📖 About

This app predicts residential house prices in Bengaluru using a machine-learning model trained on ~13,300 real property listings. Enter a few property details — location, area type, BHK, bathrooms, square footage — and get an instant price estimate along with a confidence range, based on the model's typical prediction error.

It also includes tools to explore the market: an analytics dashboard of average prices by location, a side-by-side property comparison tool, and a saved history of your past predictions.

## ✨ Features

- 🎯 **Price Prediction** — instant price estimate with a confidence band (± model MAE)
- 📊 **Market Analytics** — top locations by average price, sqft-vs-price trends
- ⚖️ **Compare Properties** — side-by-side prediction comparison for two configurations
- 🕘 **Prediction History** — saved locally, with CSV export and trend charts
- ✅ **Input Validation** — flags unrealistic inputs (e.g. too little sqft per bedroom)
- 📥 **CSV Export** — download individual predictions or full history

## 🖥️ Live Demo

**[kritikap2406-netizen-stashup-bengaluru--streamlit-appapp-k80y5b.streamlit.app](https://kritikap2406-netizen-stashup-bengaluru--streamlit-appapp-k80y5b.streamlit.app)**

## 🧠 Model

Three algorithms were trained and evaluated on the same held-out test set:

| Model | R² Score | MAE (₹ Lakh) | RMSE (₹ Lakh) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| Linear Regression | ~0.72 | ~25.4 | ~45.1 | Baseline, struggles with non-linear trends |
| Random Forest | ~0.85 | ~19.8 | ~31.2 | Captures non-linear location patterns well |
| **XGBoost** 🏆 | **~0.88** | **~18.5** | **~29.3** | **Best overall performance — used in production** |

**Understanding the metrics:**
- **R² Score** (higher is better) — how much price variance the model explains. XGBoost explains ~88% of it.
- **MAE** (lower is better) — average prediction error in Lakhs. XGBoost is off by ~₹18.5 L on average.
- **RMSE** (lower is better) — penalizes larger errors more heavily; higher than MAE here means the model occasionally misses more on extreme luxury properties, while staying accurate for typical listings.

## 🛠️ Feature Engineering

- **`bath_per_bhk`** — ratio of bathrooms to bedrooms, helping the model detect standard vs. luxury layouts
- **`sqft_per_bhk`** — average space per bedroom, used to catch unrealistic or cramped listings
- **`is_ready_to_move`** — binary flag from the "Ready to Move" input
- **Reality checks** — flags inputs outside typical Bengaluru market ranges (e.g. <300 sqft/BHK, more bathrooms than bedrooms)
- **Safety guard** — clamps any unrealistic negative/near-zero prediction to a ₹5 Lakh floor

## 📊 Exploratory Data Analysis

Univariate (price & sqft distributions), bivariate (price by area type, sqft vs. price by BHK), and a correlation heatmap — see [`Documentation/plots/`](Documentation/plots/).

## 📁 Project Structure

```
BengaluruHousePrediction/
├── Dataset/
│   └── bengaluru_house_data.csv       # 13,320 real Bengaluru listings
├── Notebook/
│   ├── bengaluru_pipeline.py          # end-to-end training pipeline
│   └── bengaluru_house_prediction.ipynb
├── Model/
│   ├── bengaluru_price_model.pkl      # trained XGBoost model
│   └── model_metadata.json
├── Streamlit_App/
│   └── app.py                         # the deployed web app
├── Documentation/
│   ├── plots/                         # EDA & model comparison charts
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## 🚀 Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/kritikap2406-netizen/STASHUP-Bengaluru_housing_price_prediction_app.git
cd STASHUP-Bengaluru_housing_price_prediction_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run the training pipeline
cd Notebook && python bengaluru_pipeline.py && cd ..

# 4. Launch the app
cd Streamlit_App && streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 📦 Dataset

Real Bengaluru house listings — 13,320 rows including `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, and `price` (in ₹ Lakh).

## 🧰 Tech Stack

Python · scikit-learn · XGBoost · Streamlit · Plotly · Pandas · Joblib

## 📌 Disclaimer

Predicted prices are **statistical estimates** based on historical data, not official valuations. Market conditions, micro-locality factors, building age, and current demand can materially affect real prices. Always consult a licensed valuer for actual transactions.

---

*Built as part of the AIML Summer Internship 2026 — Capstone Project 2, IIHMF, MNNIT Allahabad.*
