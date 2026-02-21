"""
Page 4: Modeling
Presents all ML tasks: price regression, body type classification,
market segmentation (clustering), and anomaly detection.
Loads pre-trained models and metrics for display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_engineered_data
from src.visualization import (
    apply_layout, residual_plot, predicted_vs_actual,
    feature_importance_chart, confusion_matrix_heatmap,
    elbow_plot, silhouette_plot, cluster_scatter_2d,
    COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_WARN,
)

st.set_page_config(
    page_title="Modeling | UAE Cars ML", layout="wide", page_icon="--"
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Load metrics
@st.cache_data
def load_metrics():
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        return json.load(f)

metrics = load_metrics()

st.title("Machine Learning Modeling")
st.markdown(
    "This page covers four distinct ML tasks applied to the UAE used cars dataset. "
    "Each section explains the methodology, presents the results, and interprets the "
    "findings. All models were pre-trained and their performance metrics are displayed here."
)
st.markdown("---")

# ============================================================
# Section 1: Price Prediction (Regression)
# ============================================================
st.header("Task 1: Price Prediction (Regression)")

st.markdown("""
**Objective:** Predict the listed price (AED) of a used car based on its attributes.

**Why this matters:** Price prediction helps buyers assess whether a listing is fairly priced, 
helps sellers set competitive prices, and helps dealers understand market dynamics.

**Target variable:** Price (continuous, in AED)  
**Features:** 21 features (15 numerical + 6 categorical)  
**Train/Test split:** 80% / 20% (8,000 / 2,000 records)  
**Evaluation metrics:** R-squared, RMSE, MAE, MAPE, 5-fold Cross-Validation

**Important Note on Price Distribution:** The price distribution is heavily right-skewed (skewness = 9.28) 
with values ranging from 7,183 AED to 14.7M AED. To address this, models are trained on **log-transformed 
prices** which better normalizes the target distribution. The R² values below are in log-space, which 
provides better model performance for this skewed data.
""")

# Model comparison table
st.subheader("1.1 Model Comparison")

reg_metrics = metrics["regression"]
model_names = [k for k in reg_metrics.keys() if k != "best_model"]

# Check if log-transform was used
uses_log = any(reg_metrics[m].get("uses_log_transform", False) for m in model_names if m in reg_metrics)

comparison_data = {
    "Model": model_names,
    "Train R² (log)": [reg_metrics[m]["train_r2"] for m in model_names],
    "Test R² (log)": [reg_metrics[m]["test_r2"] for m in model_names],
}

# Add original scale R² if available
if any(reg_metrics[m].get("test_r2_original_scale") is not None for m in model_names):
    comparison_data["Test R² (orig)"] = [
        reg_metrics[m].get("test_r2_original_scale", "N/A") for m in model_names
    ]

comparison_data.update({
    "RMSE (AED)": [f"{reg_metrics[m]['rmse']:,.0f}" for m in model_names],
    "MAE (AED)": [f"{reg_metrics[m]['mae']:,.0f}" for m in model_names],
    "MAPE (%)": [f"{reg_metrics[m]['mape']:.1f}" for m in model_names],
    "CV Mean R²": [reg_metrics[m]["cv_mean"] for m in model_names],
    "CV Std": [reg_metrics[m]["cv_std"] for m in model_names],
})

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

best_reg = reg_metrics["best_model"]
best_test_r2 = reg_metrics[best_reg]['test_r2']
best_orig_r2 = reg_metrics[best_reg].get('test_r2_original_scale', best_test_r2)

st.success(
    f"Best model: **{best_reg}** with Test R² = "
    f"{best_test_r2:.4f} (log-space) / {best_orig_r2:.4f} (original scale) and CV Mean = "
    f"{reg_metrics[best_reg]['cv_mean']:.4f}"
)

# Visual comparison
st.subheader("1.2 Performance Visualization")

col1, col2 = st.columns(2)

with col1:
    # R-squared comparison (train vs test)
    r2_df = pd.DataFrame({
        "Model": model_names * 2,
        "R-squared": (
            [reg_metrics[m]["train_r2"] for m in model_names] +
            [reg_metrics[m]["test_r2"] for m in model_names]
        ),
        "Set": ["Train"] * len(model_names) + ["Test"] * len(model_names),
    })
    fig = px.bar(
        r2_df, x="Model", y="R-squared", color="Set",
        barmode="group", color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY],
    )
    fig.update_layout(xaxis_tickangle=45)
    fig = apply_layout(fig, "R-squared: Train vs Test", height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # RMSE comparison
    rmse_vals = [reg_metrics[m]["rmse"] for m in model_names]
    fig = px.bar(
        x=model_names, y=rmse_vals,
        color_discrete_sequence=[COLOR_ACCENT],
        labels={"x": "Model", "y": "RMSE (AED)"},
    )
    fig.update_layout(xaxis_tickangle=45)
    fig = apply_layout(fig, "RMSE by Model (Lower is Better)", height=450)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Analysis of Results:**

- **Linear models (Linear, Ridge, Lasso)** all perform similarly with R-squared ~0.25. 
  This confirms that the relationship between features and price is substantially non-linear. 
  Despite regularization (Ridge alpha=10, Lasso alpha=100), linear models cannot capture 
  the complex interaction effects.

- **Random Forest** achieves the best test R-squared ({reg_metrics['Random Forest']['test_r2']:.4f}) 
  and the best CV score ({reg_metrics['Random Forest']['cv_mean']:.4f}). Its ensemble of decision 
  trees effectively learns the hierarchical relationship between brand, body type, and price.

- **Gradient Boosting** (Test R-sq: {reg_metrics['Gradient Boosting']['test_r2']:.4f}) 
  and **XGBoost** (Test R-sq: {reg_metrics['XGBoost']['test_r2']:.4f}) show higher training scores 
  but lower test scores than Random Forest, indicating overfitting. The boosting models tend to 
  memorize the luxury segment's extreme prices.

- **Overfitting diagnostic:** XGBoost shows the largest gap between train 
  ({reg_metrics['XGBoost']['train_r2']:.4f}) and test ({reg_metrics['XGBoost']['test_r2']:.4f}) 
  R-squared, a classic overfitting signal. Random Forest is more balanced.
  
- **Why is R-squared moderate?** The dataset has extreme price heterogeneity (10K to 15M AED) 
  driven largely by brand prestige. With ordinal encoding of 65 makes, some brand-specific 
  pricing information is lost. Target encoding or embedding-based approaches could improve this.
""")

# Predicted vs Actual
st.subheader("1.3 Best Model Deep Dive: " + best_reg)

test_results_path = os.path.join(MODELS_DIR, "regression_test_results.csv")
if os.path.exists(test_results_path):
    test_results = pd.read_csv(test_results_path)

    col1, col2 = st.columns(2)
    with col1:
        fig = predicted_vs_actual(
            test_results["Actual"], test_results["Predicted"],
            title="Predicted vs Actual Price"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = residual_plot(
            test_results["Actual"], test_results["Predicted"],
            title="Residual Plot"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Predicted vs Actual (left):** Points should cluster along the diagonal line 
    (perfect prediction). We see good prediction for cars below 500K AED, but the model 
    systematically underestimates ultra-luxury cars (>1M AED), as shown by points below the line.

    **Residual Plot (right):** Residuals should be randomly scattered around zero 
    with constant spread (homoscedasticity). Instead, we observe a fan-shaped pattern -- 
    residuals increase with predicted price -- indicating heteroscedastic errors. This 
    suggests modeling log(Price) might improve performance, or using a model with 
    variance-stabilizing properties.
    """)

# Feature Importance
st.subheader("1.4 Feature Importance")

tab_rf, tab_gb, tab_xgb = st.tabs(["Random Forest", "Gradient Boosting", "XGBoost"])

for tab, model_key in zip(
    [tab_rf, tab_gb, tab_xgb],
    ["random_forest", "gradient_boosting", "xgboost"]
):
    with tab:
        imp_path = os.path.join(MODELS_DIR, f"feature_importance_{model_key}.joblib")
        if os.path.exists(imp_path):
            feat_imp = joblib.load(imp_path)
            names = list(feat_imp.keys())
            values = list(feat_imp.values())
            fig = feature_importance_chart(
                names, values,
                title=f"Feature Importance -- {model_key.replace('_', ' ').title()}",
                height=500, top_n=21,
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Feature Importance Analysis:**

Across all three tree-based models, the most important features are consistent:

1. **Make (brand)** is consistently the top feature -- brand is the primary determinant of 
   used car prices in the UAE, more so than age or mileage
2. **Cylinders** ranks second, reflecting the engine size premium
3. **Body Type** is a strong predictor, distinguishing SUVs from sedans from sports cars
4. **Year** and **Mileage** contribute moderately -- they capture depreciation within 
   a given brand/type segment
5. **Is_Luxury** provides additional brand-level signal
6. **Location** captures geographic price differences (Dubai premium)
7. **NLP features** (has_sunroof, etc.) contribute minimally -- they were likely randomly 
   assigned in this synthetic dataset

**Practical implication:** When pricing a used car in the UAE, start with the brand and 
body type, then adjust for engine size, age, and mileage. Location matters (Dubai premium), 
but individual features and condition have minimal impact.
""")

# Why R² is limited explanation
st.subheader("1.5 Understanding Model Limitations")

st.markdown("""
**Why is R² moderate (~0.59 in log-space, ~0.43 in original scale)?**

The model's predictive power is limited by several inherent factors:

| Factor | Impact | Explanation |
|--------|--------|-------------|
| **Extreme Price Range** | High | Prices span 7K to 15M AED (2,000x ratio). Even with log-transformation, this variance is hard to capture |
| **Two-Tier Market** | High | Luxury and mass-market segments have fundamentally different pricing mechanisms |
| **Brand Encoding** | Medium | 65 unique makes are encoded ordinally, losing brand-specific premium information |
| **Missing Features** | Medium | Trim level, service history, accident details, and dealer info are not in the dataset |
| **Synthetic Data Artifacts** | Medium | Some combinations (e.g., 2005 McLaren P1) are historically impossible |

**Potential Improvements (not implemented):**

1. **Separate Models** - Train luxury and mass-market models independently
2. **Target Encoding** - Use mean price per brand instead of ordinal encoding
3. **Quantile Regression** - Model different price percentiles for robust estimates
4. **Model Stacking** - Combine multiple models with learned weights
5. **Neural Networks** - Use embeddings for categorical features
6. **More Features** - Include trim level, seller type, listing duration, etc.

**Key Insight:** The R² of ~0.59 doesn't mean the model is poor -- it reflects the inherent 
unpredictability in used car pricing. Even human experts cannot reliably predict prices with 
perfect accuracy given only these features. The model still provides valuable **relative** 
price estimates useful for deal assessment.
""")

st.markdown("---")

# ============================================================
# Section 2: Body Type Classification
# ============================================================
st.header("Task 2: Body Type Classification")

st.markdown("""
**Objective:** Predict the body type (SUV, Sedan, Coupe, etc.) of a car from its other attributes.

**Why this matters:** Body type classification can help in automated categorization of listings 
where the seller has not specified the body type, or for quality assurance to flag misclassified 
listings.

**Target variable:** Body Type (6 classes: SUV, Sedan, Coupe, Pick Up Truck, Hatchback, Sports Car)  
**Features:** Same as regression, minus Body Type  
**Train/Test split:** 80% / 20%, stratified by body type
""")

# Classification results
cls_metrics = metrics["classification"]
cls_models = [k for k in cls_metrics.keys() if k != "best_model"]

cls_comparison = pd.DataFrame({
    "Model": cls_models,
    "Accuracy": [f"{cls_metrics[m]['accuracy']:.4f}" for m in cls_models],
    "F1 (Weighted)": [f"{cls_metrics[m]['f1_weighted']:.4f}" for m in cls_models],
    "Precision (Weighted)": [f"{cls_metrics[m]['precision_weighted']:.4f}" for m in cls_models],
    "Recall (Weighted)": [f"{cls_metrics[m]['recall_weighted']:.4f}" for m in cls_models],
})
st.dataframe(cls_comparison, use_container_width=True, hide_index=True)

best_cls = cls_metrics["best_model"]
st.success(
    f"Best classifier: **{best_cls}** with Accuracy = "
    f"{cls_metrics[best_cls]['accuracy']:.4f} and F1 = "
    f"{cls_metrics[best_cls]['f1_weighted']:.4f}"
)

# Confusion Matrices
st.subheader("2.1 Confusion Matrices")

cls_labels = joblib.load(os.path.join(MODELS_DIR, "cls_labels.joblib"))

col1, col2 = st.columns(2)
for i, (model_name, col) in enumerate(zip(cls_models, [col1, col2])):
    with col:
        cm = np.array(cls_metrics[model_name]["confusion_matrix"])
        fig = confusion_matrix_heatmap(
            cm, cls_labels,
            title=model_name.replace("Classifier", "").strip(),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Confusion Matrix Analysis:**

- **SUV** is the best-predicted class (~76-80% recall) because it is the majority class 
  with distinct features (larger engines, specific brands)
- **Sedan** is reasonably well-predicted (~58-62% recall) as the second most common class
- **Coupe, Pick Up Truck, Sports Car, and Hatchback** are poorly predicted due to small 
  sample sizes and overlapping feature distributions with larger classes
- The most common misclassification is predicting minority classes as SUV or Sedan 
  (the model defaults to the majority class when uncertain)
""")

# Per-class metrics for best model
st.subheader("2.2 Per-Class Performance (" + best_cls + ")")

cls_report = cls_metrics[best_cls]["classification_report"]
class_names = [k for k in cls_report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

per_class_df = pd.DataFrame({
    "Body Type": class_names,
    "Precision": [f"{cls_report[c]['precision']:.3f}" for c in class_names],
    "Recall": [f"{cls_report[c]['recall']:.3f}" for c in class_names],
    "F1-Score": [f"{cls_report[c]['f1-score']:.3f}" for c in class_names],
    "Support": [int(cls_report[c]["support"]) for c in class_names],
})
st.dataframe(per_class_df, use_container_width=True, hide_index=True)

# Visualize per-class F1
f1_scores = [cls_report[c]["f1-score"] for c in class_names]
fig = px.bar(
    x=class_names, y=f1_scores,
    color=f1_scores,
    color_continuous_scale="RdYlGn",
    labels={"x": "Body Type", "y": "F1 Score"},
)
fig = apply_layout(fig, "F1 Score by Body Type", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Why is classification performance moderate (~61%)?**

1. **Class imbalance:** SUVs make up 46% of the dataset, creating a bias toward predicting SUV
2. **Feature overlap:** A luxury Sedan and a Coupe may have very similar price, cylinders, 
   and mileage, making them hard to distinguish from numeric features alone
3. **Missing visual features:** In reality, body type is determined by physical shape, 
   which is not captured in our feature set (no image data)
4. **Data quality:** Some body types appear mislabeled (e.g., Kia Sorento as Sedan, 
   Audi Q8 as Hatchback), adding label noise

The model reasonably separates the two dominant classes (SUV and Sedan) but struggles with 
minority classes. This is a common challenge in multi-class classification with imbalanced data.
""")

st.markdown("---")

# ============================================================
# Section 3: Market Segmentation (Clustering)
# ============================================================
st.header("Task 3: Market Segmentation (Clustering)")

st.markdown("""
**Objective:** Discover natural groupings of cars without predefined labels using unsupervised learning.

**Why this matters:** Market segmentation helps understand the structure of the used car market, 
identify target customer groups, and inform pricing strategy by segment.

**Method:** K-Means clustering with StandardScaler normalization  
**Features:** Price, Mileage, Car_Age, Cylinders, Is_Luxury  
**Optimal K selection:** Elbow method + Silhouette analysis
""")

cluster_metrics = metrics["clustering"]

# Elbow and Silhouette plots
st.subheader("3.1 Optimal K Selection")

col1, col2 = st.columns(2)
with col1:
    fig = elbow_plot(
        cluster_metrics["k_range"],
        cluster_metrics["inertias"],
        title="Elbow Method",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = silhouette_plot(
        cluster_metrics["k_range"],
        cluster_metrics["silhouettes"],
        title="Silhouette Scores",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

optimal_k = cluster_metrics["optimal_k"]
best_sil = max(cluster_metrics["silhouettes"])

st.markdown(f"""
**Selection Analysis:**
- The **Elbow Method** (left) shows a pronounced bend at K=2-3, suggesting 2-3 natural clusters
- The **Silhouette Score** (right) is highest at K={optimal_k} (score: {best_sil:.4f}), 
  indicating the clearest cluster separation at this level
- Higher K values produce diminishing returns in both metrics

**Selected K = {optimal_k}** based on the silhouette analysis, which measures how similar 
each point is to its own cluster versus neighboring clusters (range: -1 to 1, higher is better).
""")

# Cluster visualization
st.subheader("3.2 Cluster Visualization (PCA)")

pca_coords = np.array(cluster_metrics["pca_coords"])
labels = np.array(cluster_metrics["labels"])
pca_var = cluster_metrics["pca_variance"]

fig = cluster_scatter_2d(
    pca_coords[:, 0], pca_coords[:, 1], labels,
    title="K-Means Clusters in PCA Space",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
The scatter plot shows the data projected onto the first two principal components, which 
together explain **{sum(pca_var)*100:.1f}%** of the total variance 
(PC1: {pca_var[0]*100:.1f}%, PC2: {pca_var[1]*100:.1f}%). 

The two clusters are reasonably well-separated in this reduced space, with some overlap 
in the transition zone between segments.
""")

# Cluster profiles
st.subheader("3.3 Cluster Profiles")

profiles = cluster_metrics["profiles"]
profile_df = pd.DataFrame({
    "Metric": ["Count", "Avg Price (AED)", "Avg Mileage (km)", "Avg Car Age (years)",
                "Avg Cylinders", "Luxury %"],
})

for cluster_id, profile in profiles.items():
    profile_df[f"Cluster {cluster_id}"] = [
        f"{profile['count']:,}",
        f"{profile['avg_price']:,.0f}",
        f"{profile['avg_mileage']:,.0f}",
        f"{profile['avg_car_age']:.1f}",
        f"{profile['avg_cylinders']:.1f}",
        f"{profile['luxury_pct']:.1f}%",
    ]

st.dataframe(profile_df, use_container_width=True, hide_index=True)

# Segment interpretation
if optimal_k == 2:
    p0 = profiles["0"]
    p1 = profiles["1"]

    # Determine which is luxury
    luxury_cluster = "0" if p0["luxury_pct"] > p1["luxury_pct"] else "1"
    mass_cluster = "1" if luxury_cluster == "0" else "0"

    st.markdown(f"""
**Segment Interpretation:**

| Attribute | Cluster {mass_cluster} -- Mass Market | Cluster {luxury_cluster} -- Luxury |
|---|---|---|
| **Size** | {profiles[mass_cluster]['count']:,} cars ({profiles[mass_cluster]['count']/10000*100:.0f}%) | {profiles[luxury_cluster]['count']:,} cars ({profiles[luxury_cluster]['count']/10000*100:.0f}%) |
| **Avg Price** | {profiles[mass_cluster]['avg_price']:,.0f} AED | {profiles[luxury_cluster]['avg_price']:,.0f} AED |
| **Avg Cylinders** | {profiles[mass_cluster]['avg_cylinders']:.1f} | {profiles[luxury_cluster]['avg_cylinders']:.1f} |
| **Luxury Brands** | {profiles[mass_cluster]['luxury_pct']:.1f}% | {profiles[luxury_cluster]['luxury_pct']:.1f}% |

The clustering reveals a **two-tier market structure**:

- **Mass Market Segment** ({profiles[mass_cluster]['count']/10000*100:.0f}%): Characterized by 
  lower prices (~{profiles[mass_cluster]['avg_price']:,.0f} AED), 
  ~{profiles[mass_cluster]['avg_cylinders']:.0f}-cylinder engines, and virtually no luxury brands. 
  These are everyday vehicles for commuting, family use, and practical transportation.

- **Luxury Segment** ({profiles[luxury_cluster]['count']/10000*100:.0f}%): Characterized by 
  {profiles[luxury_cluster]['avg_price']/profiles[mass_cluster]['avg_price']:.1f}x higher prices, 
  larger engines (~{profiles[luxury_cluster]['avg_cylinders']:.0f} cylinders), and 
  {profiles[luxury_cluster]['luxury_pct']:.0f}% luxury brand composition. 
  This segment represents high-net-worth buyers seeking premium vehicles.

Both segments have similar mileage and car age distributions, confirming that the segmentation 
is driven by price and brand rather than vehicle usage patterns.
    """)
else:
    st.markdown("""
The clustering reveals multiple market segments with distinct price, mileage, and brand 
profiles. Each segment represents a different buyer persona in the UAE used car market.
    """)

st.markdown("---")

# ============================================================
# Section 4: Anomaly Detection
# ============================================================
st.header("Task 4: Anomaly Detection")

st.markdown("""
**Objective:** Identify unusual listings that deviate significantly from market norms -- 
potentially overpriced, underpriced, or otherwise suspicious cars.

**Why this matters:** Anomaly detection helps buyers spot potential deals (underpriced cars) 
or avoid overpriced listings. For marketplaces, it flags potential fraudulent or erroneous listings.

**Method:** Isolation Forest (unsupervised)  
**Features:** Price, Mileage, Car_Age, Cylinders, Is_Luxury  
**Contamination rate:** 5% (expect 5% of data to be anomalous)
""")

# Anomaly summary
anomaly_metrics = metrics["anomaly_detection"]
col1, col2, col3 = st.columns(3)
col1.metric("Anomalies Detected", f"{anomaly_metrics['n_anomalies']}")
col2.metric("Anomaly Rate", f"{anomaly_metrics['pct_anomalies']}%")
col3.metric("Normal Listings", f"{10000 - anomaly_metrics['n_anomalies']:,}")

# Load anomaly results
anomaly_path = os.path.join(MODELS_DIR, "anomaly_results.csv")
if os.path.exists(anomaly_path):
    df_anom = pd.read_csv(anomaly_path)

    st.subheader("4.1 Anomaly Visualization")

    # Scatter: Price vs Mileage, colored by anomaly
    df_anom["Status"] = df_anom["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    fig = px.scatter(
        df_anom, x="Mileage", y="Price",
        color="Status",
        color_discrete_map={"Normal": COLOR_PRIMARY, "Anomaly": COLOR_WARN},
        opacity=0.5,
        hover_data=["Make", "Model", "Year"],
    )
    fig = apply_layout(fig, "Price vs Mileage -- Anomalies Highlighted", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The scatter plot highlights anomalous listings in red. Most anomalies cluster in two areas:
    
    1. **High-price outliers:** Luxury/supercar listings with prices >1M AED that stand out 
       from the mass market
    2. **Unusual combinations:** Cars where the price-mileage-age-cylinder combination is 
       statistically unusual (e.g., a high-mileage luxury car, or a low-mileage car with 
       an unexpectedly high or low price)
    """)

    # Top anomalies table
    st.subheader("4.2 Most Anomalous Listings")

    top_anom = df_anom[df_anom["Is_Anomaly"]].nsmallest(15, "Anomaly_Score").copy()
    top_anom["Price_Display"] = top_anom["Price"].apply(lambda x: f"{x:,.0f} AED")
    top_anom["Mileage_Display"] = top_anom["Mileage"].apply(lambda x: f"{x:,.0f} km")
    top_anom["Anomaly_Score"] = top_anom["Anomaly_Score"].round(4)

    display_cols = ["Make", "Model", "Year", "Price_Display", "Mileage_Display",
                    "Body Type", "Location", "Anomaly_Score"]
    st.dataframe(
        top_anom[display_cols].rename(columns={
            "Price_Display": "Price", "Mileage_Display": "Mileage"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    **How to read Anomaly Scores:**
    - Scores closer to **-1** are the most anomalous (highly unusual)
    - Scores closer to **0** are borderline (marginally unusual)
    - Scores above **0** are normal (typical market listings)
    
    The Isolation Forest algorithm works by randomly partitioning data points. Anomalies are 
    isolated quickly (few partitions needed), while normal points require many splits. The 
    anomaly score reflects this isolation depth.
    """)

    # Anomaly score distribution
    st.subheader("4.3 Anomaly Score Distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_anom[~df_anom["Is_Anomaly"]]["Anomaly_Score"],
        name="Normal", marker_color=COLOR_PRIMARY, opacity=0.7, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=df_anom[df_anom["Is_Anomaly"]]["Anomaly_Score"],
        name="Anomaly", marker_color=COLOR_WARN, opacity=0.7, nbinsx=30,
    ))
    fig.update_layout(barmode="overlay", xaxis_title="Anomaly Score", yaxis_title="Count")
    fig = apply_layout(fig, "Distribution of Anomaly Scores", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The bimodal distribution shows clear separation between normal listings (scores > 0) 
    and anomalies (scores < 0). The threshold is determined automatically by the Isolation 
    Forest at the configured contamination rate of 5%.
    """)

st.markdown("---")

# ============================================================
# Section 5: Methodology Notes
# ============================================================
st.header("Methodology Notes")

st.markdown("""
### Modeling Decisions and Tradeoffs

**Data Split Strategy:**
- Regression: Random 80/20 split (no stratification needed for continuous target)
- Classification: Stratified 80/20 split to preserve class proportions in both sets
- All models use `random_state=42` for reproducibility

**Hyperparameter Choices:**
- Random Forest: 200 trees, max_depth=20, min_samples_split=5
- Gradient Boosting: 200 trees, max_depth=6, learning_rate=0.1
- XGBoost: 300 trees, max_depth=7, learning_rate=0.1, subsample=0.8
- These were chosen based on common best practices for datasets of this size (~10K records)

**What could improve the models:**
1. **Target encoding** for Make and Model would better capture brand-price relationships 
   than ordinal encoding
2. **Log-transformed price** as the target variable would address heteroscedasticity 
   and improve predictions for extreme values
3. **Hyperparameter tuning** via GridSearchCV or Bayesian optimization could find better 
   configurations (not done here to keep training time manageable)
4. **Stacking/ensembling** multiple models could capture different aspects of the data
5. **Feature interactions** (e.g., Make x Body_Type, Year x Is_Luxury) could improve 
   linear model performance
6. **Handling the synthetic nature** of the data: in a real-world scenario, genuine 
   market data would likely yield better model performance due to more realistic 
   price-feature relationships
""")
