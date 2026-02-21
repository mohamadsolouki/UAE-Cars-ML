"""
Page 6: About
Project documentation, methodology summary, tech stack,
limitations, and future work.
"""

import streamlit as st
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

st.set_page_config(
    page_title="About | UAE Cars ML", layout="wide", page_icon="--"
)

st.title("About This Project")
st.markdown("---")

# ============================================================
# Section 1: Project Overview
# ============================================================
st.header("1. Project Overview")

st.markdown("""
This project is a comprehensive, end-to-end machine learning application built on a dataset 
of 10,000 used car listings from the UAE market. It demonstrates the complete data science 
lifecycle from data exploration to model deployment, and is designed to serve as educational 
material for aspiring data scientists and ML practitioners.

**Project Goals:**
- Demonstrate the full ML pipeline: data exploration, cleaning, feature engineering, 
  modeling, evaluation, and prediction
- Provide insightful analysis of the UAE used car market
- Build practical, interactive tools for price estimation and deal assessment
- Serve as a learning resource with detailed explanations of every step
""")

st.markdown("---")

# ============================================================
# Section 2: Dataset
# ============================================================
st.header("2. Dataset")

st.markdown("""
**Source:** UAE Used Cars 10K dataset  
**Size:** 10,001 rows, 12 columns  
**Coverage:** 65 car manufacturers, 488 models, 8 emirates, years 2005-2024

**Columns:**

| Column | Type | Description |
|---|---|---|
| Make | Categorical | Car manufacturer (65 unique) |
| Model | Categorical | Car model (488 unique) |
| Year | Numerical | Manufacturing year (2005-2024) |
| Price | Numerical | Listed price in AED |
| Mileage | Numerical | Odometer reading in km |
| Body Type | Categorical | Vehicle classification (10 types) |
| Cylinders | Mixed | Engine cylinder count (contains "Unknown") |
| Transmission | Categorical | Automatic or Manual |
| Fuel Type | Categorical | Gasoline, Diesel, Electric, Hybrid |
| Color | Categorical | Exterior color (18 colors) |
| Location | Categorical | UAE emirate (8 locations) |
| Description | Free Text | Semi-structured text with features and condition |

**Data Quality Notes:**
- The dataset appears to be synthetically generated, based on uniform distributions for 
  year and mileage, templated descriptions, and some impossible year-model combinations
- Despite being synthetic, it realistically represents the structure of the UAE car market 
  with appropriate brand distributions, price ranges, and geographic patterns
- This makes it ideal for educational purposes while requiring the same processing steps 
  as real-world data
""")

st.markdown("---")

# ============================================================
# Section 3: Methodology
# ============================================================
st.header("3. Methodology")

st.markdown("""
### Data Processing Pipeline

1. **Cleaning:** Standardized whitespace in Location, converted Cylinders to numeric 
   (imputing "Unknown" via Make-based median), created display-friendly name versions
2. **NLP Extraction:** Used regex to parse free-text Description into structured Condition 
   and binary feature indicator columns
3. **Feature Engineering:** Created 9 derived features including Car_Age, Mileage_Per_Year, 
   Log transformations, Condition_Score, Is_Luxury, and Feature_Count
4. **Encoding:** Ordinal encoding for categorical features, StandardScaler for numerical 
   features, applied via scikit-learn ColumnTransformer

### Machine Learning Tasks

| Task | Type | Best Model | Key Metric |
|---|---|---|---|
| Price Prediction | Regression | Random Forest | R-squared: 0.55, CV: 0.58 |
| Body Type Classification | Multi-class | Gradient Boosting | F1 (weighted): 0.60 |
| Market Segmentation | Clustering | K-Means (K=2) | Silhouette: 0.48 |
| Anomaly Detection | Unsupervised | Isolation Forest | 5% contamination |

### Evaluation Strategy
- **Regression:** 80/20 train-test split, 5-fold cross-validation, multiple metrics 
  (R-squared, RMSE, MAE, MAPE)
- **Classification:** Stratified 80/20 split, confusion matrix analysis, per-class metrics
- **Clustering:** Elbow method + silhouette analysis for K selection
- **Anomaly Detection:** Isolation Forest with 5% contamination rate, visual inspection of results
""")

st.markdown("---")

# ============================================================
# Section 4: Performance Summary
# ============================================================
st.header("4. Model Performance Summary")

# Load metrics
with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
    all_metrics = json.load(f)

reg = all_metrics["regression"]

st.subheader("Regression Models")
reg_models = [k for k in reg if k != "best_model"]
reg_df_data = []
for m in reg_models:
    reg_df_data.append({
        "Model": m,
        "Test R-squared": reg[m]["test_r2"],
        "CV Mean R-squared": reg[m]["cv_mean"],
        "RMSE (AED)": f"{reg[m]['rmse']:,.0f}",
        "MAE (AED)": f"{reg[m]['mae']:,.0f}",
    })
st.dataframe(
    reg_df_data,
    use_container_width=True,
    hide_index=True,
)

cls = all_metrics["classification"]
st.subheader("Classification Models")
cls_models = [k for k in cls if k != "best_model"]
cls_df_data = []
for m in cls_models:
    cls_df_data.append({
        "Model": m,
        "Accuracy": cls[m]["accuracy"],
        "F1 (Weighted)": cls[m]["f1_weighted"],
        "Precision (Weighted)": cls[m]["precision_weighted"],
        "Recall (Weighted)": cls[m]["recall_weighted"],
    })
st.dataframe(
    cls_df_data,
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ============================================================
# Section 5: Tech Stack
# ============================================================
st.header("5. Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Processing")
    st.markdown("""
    - **pandas** -- Data manipulation and analysis
    - **NumPy** -- Numerical computing
    - **regex** -- Text parsing and NLP extraction
    """)

with col2:
    st.subheader("Machine Learning")
    st.markdown("""
    - **scikit-learn** -- Core ML framework (preprocessing, models, evaluation)
    - **XGBoost** -- Gradient boosted trees for regression
    - **joblib** -- Model serialization and caching
    """)

with col3:
    st.subheader("Visualization and App")
    st.markdown("""
    - **Streamlit** -- Web application framework
    - **Plotly** -- Interactive charts and visualizations
    - **Streamlit Community Cloud** -- Deployment platform
    """)

st.markdown("---")

# ============================================================
# Section 6: Limitations
# ============================================================
st.header("6. Limitations and Caveats")

st.markdown("""
### Data Limitations
- **Synthetic data:** The dataset appears to be synthetically generated, which means 
  patterns may not fully reflect real-world UAE market dynamics
- **No image data:** Body type and condition would benefit from visual inspection
- **No trim/variant info:** A BMW 3-Series 320i and M340i have very different prices, 
  but both appear as "3-series"
- **Static snapshot:** The data represents a single point in time; market prices fluctuate

### Model Limitations
- **Moderate accuracy:** The best regression model achieves R-squared of ~0.55, meaning 
  45% of price variance is unexplained -- likely due to missing trim/variant information
- **Ordinal encoding limitations:** Treating Make as ordinal imposes an arbitrary ordering; 
  target encoding would better capture brand-price relationships
- **Class imbalance:** Body type classification is biased toward SUV and Sedan predictions
- **No temporal validation:** Models were not tested on future data (time-based split)

### Application Limitations
- Predictions should not be used as the sole basis for buying/selling decisions
- The deal analyzer provides estimates based on historical patterns, not real-time market data
- Model performance may degrade for unusual configurations not well-represented in training data
""")

st.markdown("---")

# ============================================================
# Section 7: Future Work
# ============================================================
st.header("7. Future Improvements")

st.markdown("""
### Short-term Improvements
- **Target encoding** for Make and Model to better capture brand-price relationships
- **Hyperparameter optimization** using Bayesian optimization (Optuna or similar)
- **Log-price regression** to address heteroscedasticity and improve luxury car predictions
- **SMOTE or class weighting** to address body type classification imbalance

### Medium-term Enhancements
- **Real-world data integration** from actual UAE car listing platforms
- **Time series analysis** to capture price trends and seasonal patterns
- **Deep learning** using neural network embeddings for categorical features
- **SHAP analysis** for detailed model interpretability and explanations

### Long-term Vision
- **Image-based classification** using computer vision for body type and condition assessment
- **Real-time pricing API** that updates with market conditions
- **Price trend forecasting** using historical pricing data
- **Recommendation engine** suggesting similar vehicles based on user preferences
""")

st.markdown("---")

# ============================================================
# Section 8: How to Use
# ============================================================
st.header("8. How to Run Locally")

st.markdown("""
### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/mohamadsolouki/UAE-Cars-ML.git
cd UAE-Cars-ML

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models
python train.py

# Launch the app
streamlit run app.py
```

### Project Structure

```
UAE-Cars-ML/
    .streamlit/
        config.toml          # Streamlit theme configuration
    src/
        __init__.py
        data_loader.py       # Data loading with caching
        preprocessing.py     # Cleaning and text extraction
        feature_engineering.py  # Feature transformations
        visualization.py     # Reusable Plotly chart functions
    pages/
        1_Data_Overview.py   # Dataset exploration
        2_Exploratory_Analysis.py  # Visual EDA
        3_Feature_Engineering.py   # Feature transformations
        4_Modeling.py        # ML model results
        5_Predictions.py     # Interactive prediction tools
        6_About.py           # Project documentation
    models/                  # Pre-trained model artifacts
    app.py                   # Main application entry point
    train.py                 # Model training script
    requirements.txt         # Python dependencies
    uae_used_cars_10k.csv    # Dataset
```
""")

st.markdown("---")

st.markdown("""
### License

This project is licensed under the **MIT License**. See the LICENSE file for details.

### Author

Mohamad Solouki -- 2026

Built with Streamlit, scikit-learn, XGBoost, and Plotly.
""")
