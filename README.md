# STASHUP: A Bengaluru House Price Prediction System

AIML Summer Internship 2026 — Capstone Project 2 (IIHMF, MNNIT Allahabad)

## Dataset
Real Bengaluru house listings, 13,320 rows: `area_type`, `availability`, `location`,
`size`, `society`, `total_sqft`, `bath`, `balcony`, `price` (in Lakh INR).

## Folder Structure
```
BengaluruHousePrediction/
├── Dataset/
│   └── bengaluru_house_data.csv
├── Notebook/
│   ├── bengaluru_pipeline.py
│   └── bengaluru_house_prediction.ipynb
├── Model/
│   ├── bengaluru_price_model.pkl
│   └── model_metadata.json
├── Streamlit_App/
│   └── app.py
├── Documentation/
│   ├── plots/                 # 6 EDA + comparison charts
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
cd Notebook && python bengaluru_pipeline.py && cd ..
cd Streamlit_App && streamlit run app.py
```

## Phase-by-phase notes

**Phase 3 — Data Preprocessing**

## 🛠️ Data Processing

Before the machine learning model can estimate a house price, your inputs go through a quick and smart preparation phase. Here’s what happens behind the scenes:

* **Feature Engineering:** Instead of just passing raw numbers to the model, the app automatically calculates two helpful ratios: **Bathrooms per BHK** (`bath_per_bhk`) and **Square Footage per BHK** (`sqft_per_bhk`). This helps the model understand the layout efficiency of the property.
* **Categorical Formatting:** Text inputs like *Location* and *Area Type* are matched against the exact categories the model was trained on. The "Ready to Move" checkbox is converted into a binary format (1 or 0) so the model can process it.
* **Reality Checks (Validation):** Before predicting, the app runs a sanity check on your inputs. If you enter something unusual for the Bengaluru market—like less than 300 sqft for a 3 BHK, or more bathrooms than bedrooms—the app flags a mild warning. This ensures you get alerted if your inputs fall outside typical real-world ranges. 
* **Safety Guards:** If the model outputs an unrealistic negative or near-zero price due to extreme inputs, the app automatically clamps the minimum predicted value to ₹5 Lakh to keep the estimates sensible.

**Phase 4 — EDA**
Univariate (price, sqft histograms), bivariate (price by area type boxplot,
sqft vs price scatter colored by BHK), and a correlation heatmap — see
`Documentation/plots/`.

**Phase 5 
## ⚙️ Feature Engineering

To help the model make smarter predictions, we automatically create a few new data points from your inputs:

* **`bath_per_bhk`**: Calculates the ratio of bathrooms to bedrooms. This helps the model gauge if the property has a standard or luxury layout.
* **`sqft_per_bhk`**: Calculates the average space per bedroom. This is a crucial metric to identify cramped properties or overly spacious luxury homes.
* **`is_ready_to_move`**: Converts the "Ready to Move" checkbox into a machine-readable format (1 for Yes, 0 for No).

**Phase 6/7 — Models & Evaluation**

## 📊 Model Evaluation

We tested three different machine learning algorithms to find the best predictor for Bengaluru house prices. All models were evaluated on the same unseen test dataset.

| Model | R² Score | MAE (₹ Lakh) | RMSE (₹ Lakh) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | ~0.72 | ~25.4 | ~45.1 | Baseline model, struggles with complex non-linear trends |
| **Random Forest** | ~0.85 | ~19.8 | ~31.2 | Good at capturing non-linear location patterns |
| **XGBoost** 🏆 | ~0.88 | ~18.5 | ~29.3 | **Best overall performance** |

### 💡 Understanding the Metrics
* **R² Score (higher is better):** Represents how well the model explains price variations. XGBoost explains ~88% of the variance.
* **MAE - Mean Absolute Error (lower is better):** The average error in Lakhs. On average, XGBoost predictions are off by ~₹18.5 Lakhs.
* **RMSE - Root Mean Squared Error (lower is better):** Penalizes large errors. Since RMSE is higher than MAE, all models make a few larger mistakes on extreme luxury properties, but are highly accurate for standard apartments.



**Phase 8 — Deployment**
`Streamlit_App/app.py` takes location, area type, BHK, bathrooms, balconies,
sqft, and ready-to-move status, and returns a live price prediction with the
model's typical error margin shown alongside it.
