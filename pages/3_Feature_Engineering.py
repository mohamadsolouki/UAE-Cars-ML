"""
Page 3: Feature Engineering
Demonstrates the transformation of raw data into model-ready features,
with explanations of each step and before/after comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_raw_data, load_clean_data, load_engineered_data
from src.preprocessing import ALL_FEATURES, CONDITION_ORDER
from src.feature_engineering import get_model_features, get_categorical_features
from src.visualization import (
    correlation_heatmap, bar_chart, histogram, scatter_plot,
    apply_layout, COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT,
)

st.set_page_config(
    page_title="Feature Engineering | UAE Cars ML", layout="wide", page_icon="--"
)

st.title("Feature Engineering")
st.markdown(
    "Feature engineering is the process of transforming raw data into meaningful inputs "
    "for machine learning models. Well-engineered features can dramatically improve model "
    "performance. This page walks through every transformation applied to the dataset, "
    "explaining the rationale behind each decision."
)
st.markdown("---")

df_raw = load_raw_data()
df_clean = load_clean_data()
df_eng = load_engineered_data()

# ============================================================
# Section 1: Overview of Transformations
# ============================================================
st.header("1. Transformation Pipeline Overview")

st.markdown("""
The feature engineering pipeline transforms the original 12 columns into a richer set of 
features designed for machine learning. The pipeline follows these stages:

| Stage | Description | Columns Added |
|---|---|---|
| **Cleaning** | Strip whitespace, convert types, impute missing values | 3 (Make_Display, Model_Display, Condition) |
| **NLP Extraction** | Parse Description into structured features | 7 (6 binary + Features_List) |
| **Numerical Derivation** | Create computed features from existing ones | 6 (Car_Age, Mileage_Per_Year, Log_Price, Log_Mileage, Price_Per_Km, Condition_Score) |
| **Domain Features** | Encode domain knowledge (luxury brands, feature count) | 2 (Is_Luxury, Feature_Count) |
| **Encoding** | Transform categoricals for model consumption | Applied at training time |

**Result:** From 12 raw columns to 21+ model-ready features.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Original Columns", "12")
col2.metric("Engineered Features", f"{len(get_model_features()) + len(get_categorical_features())}")
col3.metric("Total After Engineering", f"{len(df_eng.columns)}")

st.markdown("---")

# ============================================================
# Section 2: Text Feature Extraction (NLP)
# ============================================================
st.header("2. Text Feature Extraction from Description")

st.markdown("""
The Description column contains semi-structured text following a consistent template. 
Rather than discarding this rich information, we extract two types of structured data:

1. **Condition** -- The overall vehicle condition (e.g., "No damage", "Accident history")
2. **Car Features** -- Individual amenities (e.g., "Sunroof", "Leather seats")

This is a practical example of **Natural Language Processing (NLP)** applied to real-world data, 
using regex-based pattern matching rather than complex language models.
""")

# Show before and after
st.subheader("2.1 Extraction Demo")
st.markdown("**Raw Description Text:**")
sample_idx = 0
st.code(df_raw["Description"].iloc[sample_idx], language=None)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Extracted Condition:**")
    st.info(df_clean["Condition"].iloc[sample_idx])
with col2:
    st.markdown("**Extracted Features:**")
    features = df_clean["Features_List"].iloc[sample_idx]
    if features:
        for f in features:
            st.write(f"- {f}")
    else:
        st.write("None found")

# Feature frequency
st.subheader("2.2 Feature Frequency Analysis")
st.markdown(
    "Each extracted feature is converted to a binary indicator column (0 or 1). "
    "Here is how frequently each feature appears across the dataset:"
)

feature_freq = {}
for feat in ALL_FEATURES:
    col_name = "has_" + feat.lower().replace(" ", "_")
    feature_freq[feat] = df_eng[col_name].sum()

freq_df = pd.DataFrame({
    "Feature": list(feature_freq.keys()),
    "Count": list(feature_freq.values()),
    "Percentage": [v / len(df_eng) * 100 for v in feature_freq.values()],
}).sort_values("Count", ascending=True)

fig = px.bar(
    freq_df, x="Count", y="Feature", orientation="h",
    text=[f"{p:.1f}%" for p in freq_df["Percentage"]],
    color_discrete_sequence=[COLOR_PRIMARY],
)
fig.update_traces(textposition="outside")
fig = apply_layout(fig, "Car Feature Frequency in Dataset", height=350)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
All 6 features appear in roughly similar proportions (48-56% of listings), suggesting they were 
randomly assigned during data generation. In a real dataset, we would expect certain features 
(e.g., Bluetooth) to be far more common than luxury additions (e.g., Adaptive cruise control). 
Despite this, the binary indicators still provide useful signal for modeling.
""")

# Condition encoding
st.subheader("2.3 Condition Severity Encoding")
st.markdown("""
The Condition field is an ordinal variable -- conditions range from "No damage" (best) to 
"Accident history" (most severe). We encode this as a numeric severity score:
""")

cond_df = pd.DataFrame({
    "Condition": list(CONDITION_ORDER.keys()),
    "Severity Score": list(CONDITION_ORDER.values()),
    "Interpretation": [
        "Vehicle has no cosmetic or mechanical issues",
        "Minor surface-level scratches only",
        "Bumper has been repainted (cosmetic repair)",
        "One or more doors have dents",
        "Engine has undergone repair work",
        "Vehicle has been in an accident",
        "Condition information not available",
    ]
})
st.dataframe(cond_df, use_container_width=True, hide_index=True)

st.markdown("""
**Why ordinal encoding?** Unlike one-hot encoding, ordinal encoding preserves the inherent 
ordering of severity levels. This allows models to learn that "Accident history" (score=5) is 
worse than "Minor scratches" (score=1), which one-hot encoding would not capture.
""")

st.markdown("---")

# ============================================================
# Section 3: Derived Numerical Features
# ============================================================
st.header("3. Derived Numerical Features")

st.markdown("""
Creating new features from existing columns can reveal hidden patterns. 
Here are the engineered numerical features and their rationale:
""")

# Car Age
st.subheader("3.1 Car Age")
st.markdown("""
**Formula:** `Car_Age = 2025 - Year`  
**Rationale:** Age is more intuitive than year for depreciation modeling. 
A 2020 car is 5 years old regardless of when the analysis is done (given a fixed reference year). 
Values are clipped to a minimum of 1 to avoid division-by-zero issues.
""")

col1, col2 = st.columns(2)
with col1:
    fig = histogram(df_eng, "Car_Age", title="Distribution of Car Age (Years)", nbins=20, color=COLOR_PRIMARY)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = scatter_plot(df_eng, "Car_Age", "Price", title="Price vs Car Age", trendline="lowess")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
The Car Age distribution is uniform (1-20 years), mirroring the Year distribution. 
The scatter plot shows a very slight downward trend in the LOWESS fit, 
but the relationship is weak due to the confounding effect of brand and body type.
""")

# Mileage Per Year
st.subheader("3.2 Mileage Per Year")
st.markdown("""
**Formula:** `Mileage_Per_Year = Mileage / Car_Age`  
**Rationale:** This normalizes mileage by age, creating a "driving intensity" metric. 
A car with 100K km over 10 years is used differently than one with 100K km over 2 years. 
High mileage-per-year might indicate commercial use or long-distance commuting.
""")

col1, col2 = st.columns(2)
with col1:
    fig = histogram(df_eng, "Mileage_Per_Year", title="Mileage Per Year Distribution", nbins=60)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = scatter_plot(df_eng, "Mileage_Per_Year", "Price", title="Price vs Mileage Per Year", trendline="lowess")
    st.plotly_chart(fig, use_container_width=True)

# Log transforms
st.subheader("3.3 Log Transformations")
st.markdown("""
**Formula:** `Log_Price = log(1 + Price)`, `Log_Mileage = log(1 + Mileage)`  
**Rationale:** Price spans from ~10K to ~15M AED -- a 1,500x range. Log transformation 
compresses this range, reducing the influence of extreme values and making the distribution 
more symmetric. We use `log1p` (natural log of 1+x) to handle edge cases where values might be zero.
""")

col1, col2 = st.columns(2)
with col1:
    fig = histogram(df_eng, "Price", title="Price (Original Scale)", nbins=60, color=COLOR_SECONDARY)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = histogram(df_eng, "Log_Price", title="Price (Log Scale)", nbins=60, color=COLOR_ACCENT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
The log transformation shifts the distribution from extreme right-skew to an approximately 
bell-shaped curve. This is beneficial for linear models that assume normally distributed errors, 
and helps tree-based models by reducing the impact of extreme outliers.
""")

# Luxury indicator
st.subheader("3.4 Luxury Brand Indicator")
st.markdown("""
**Logic:** `Is_Luxury = 1` if Make is one of: Ferrari, Lamborghini, Rolls-Royce, Bentley, 
Maserati, Aston Martin, McLaren, Maybach, Porsche, Bugatti  
**Rationale:** These brands operate in a fundamentally different price segment. 
The binary indicator captures this domain knowledge directly, saving the model from 
having to infer it from the Make encoding alone.
""")

luxury_stats = df_eng.groupby("Is_Luxury").agg(
    Count=("Price", "count"),
    Avg_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Avg_Cylinders=("Cylinders", "mean"),
).reset_index()
luxury_stats["Is_Luxury"] = luxury_stats["Is_Luxury"].map({0: "Non-Luxury", 1: "Luxury"})
luxury_stats["Avg_Price"] = luxury_stats["Avg_Price"].apply(lambda x: f"{x:,.0f} AED")
luxury_stats["Median_Price"] = luxury_stats["Median_Price"].apply(lambda x: f"{x:,.0f} AED")
luxury_stats["Avg_Cylinders"] = luxury_stats["Avg_Cylinders"].round(1)

st.dataframe(luxury_stats, use_container_width=True, hide_index=True)

# Feature Count
st.subheader("3.5 Feature Count")
st.markdown("""
**Formula:** `Feature_Count = number of extracted features per listing`  
**Rationale:** Cars with more features (sunroof + leather + navigation + ...) tend to be 
higher-trim models with higher prices. This single numeric feature captures the overall 
"feature richness" of each listing.
""")

col1, col2 = st.columns(2)
with col1:
    fc_dist = df_eng["Feature_Count"].value_counts().sort_index().reset_index()
    fc_dist.columns = ["Feature_Count", "Listings"]
    fig = px.bar(fc_dist, x="Feature_Count", y="Listings", color_discrete_sequence=[COLOR_PRIMARY])
    fig = apply_layout(fig, "Distribution of Feature Count", height=350)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fc_price = df_eng.groupby("Feature_Count")["Price"].mean().reset_index()
    fig = px.bar(fc_price, x="Feature_Count", y="Price", color_discrete_sequence=[COLOR_ACCENT])
    fig.update_layout(yaxis_title="Average Price (AED)")
    fig = apply_layout(fig, "Average Price by Feature Count", height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================
# Section 4: Categorical Encoding Strategy
# ============================================================
st.header("4. Categorical Encoding Strategy")

st.markdown("""
Machine learning models require numeric inputs. Converting categorical variables to numbers 
is a critical design decision that affects model performance. Here is our encoding strategy:
""")

encoding_strategy = pd.DataFrame({
    "Feature": ["Make", "Model", "Body Type", "Transmission", "Fuel Type", "Color", "Location"],
    "Unique Values": [
        df_eng["Make"].nunique(), df_eng["Model"].nunique(),
        df_eng["Body Type"].nunique(), df_eng["Transmission"].nunique(),
        df_eng["Fuel Type"].nunique(), df_eng["Color"].nunique(),
        df_eng["Location"].nunique(),
    ],
    "Cardinality": ["High (65)", "Very High (488)", "Medium (10)", "Low (2)", "Low (4)", "Medium (18)", "Low (8)"],
    "Encoding Method": [
        "Ordinal Encoding",
        "Excluded (captured by Make + other features)",
        "Ordinal Encoding",
        "Ordinal Encoding",
        "Ordinal Encoding",
        "Ordinal Encoding",
        "Ordinal Encoding",
    ],
    "Rationale": [
        "Tree-based models handle ordinal well; combined with Is_Luxury flag",
        "Too high cardinality; brand + body type + specs capture most info",
        "Natural groupings; tree models split effectively on ordinal values",
        "Only 2 values; simple encoding sufficient",
        "Few unique values; ordinal encoding is compact",
        "Tree-based models handle medium cardinality well",
        "Geographic feature; ordinal works for tree-based models",
    ],
})
st.dataframe(encoding_strategy, use_container_width=True, hide_index=True)

st.markdown("""
**Why Ordinal Encoding over One-Hot?**

For tree-based models (Random Forest, Gradient Boosting, XGBoost), ordinal encoding is often 
preferred because:

1. **Dimensionality:** One-hot encoding Make (65 values) would add 65 columns. Ordinal keeps it at 1.
2. **Tree splits:** Trees split on thresholds, so they can still isolate individual categories.
3. **Efficiency:** Fewer columns means faster training and less risk of the curse of dimensionality.
4. **Unknown handling:** Ordinal encoding can map unseen categories to a default value (-1).

For linear models, this encoding is less ideal (it assumes an artificial ordering), which is 
one reason why tree-based models outperform linear ones on this dataset.
""")

st.markdown("---")

# ============================================================
# Section 5: Feature Correlation Analysis
# ============================================================
st.header("5. Feature Correlation Analysis")

st.markdown("""
Before modeling, we examine how features relate to each other and to the target variable (Price). 
High correlation between features (multicollinearity) can cause issues for linear models, 
while correlation with the target indicates predictive power.
""")

# Correlation with Price
st.subheader("5.1 Feature Correlation with Price")

num_features = [
    "Cylinders", "Is_Luxury", "Feature_Count", "Condition_Score",
    "Car_Age", "Mileage_Per_Year", "Mileage", "Year",
    "has_sunroof", "has_leather_seats", "has_navigation_system",
    "has_bluetooth", "has_rear_camera", "has_adaptive_cruise_control",
]

price_corr = df_eng[num_features + ["Price"]].corr()["Price"].drop("Price").sort_values()

fig = go.Figure()
colors = [COLOR_ACCENT if v > 0 else COLOR_SECONDARY for v in price_corr.values]
fig.add_trace(go.Bar(
    x=price_corr.values,
    y=price_corr.index,
    orientation="h",
    marker_color=colors,
    text=[f"{v:.3f}" for v in price_corr.values],
    textposition="outside",
))
fig.update_layout(xaxis_title="Pearson Correlation with Price", yaxis_title="")
fig = apply_layout(fig, "Feature Correlation with Price", height=550)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Analysis:**
- **Cylinders (0.45)** is the strongest linear predictor -- more cylinders means more powerful, 
  more expensive vehicles
- **Is_Luxury (0.39)** captures brand prestige, the second strongest predictor
- **Feature Count (0.05)** shows weak positive correlation -- more features slightly increase price
- **Condition Score, Car Age, Mileage** all show near-zero correlation, suggesting non-linear 
  relationships or confounding by other variables
- The binary feature indicators (sunroof, leather, etc.) show minimal individual correlation, 
  but collectively may contribute signal through Feature_Count
""")

# Full correlation matrix
st.subheader("5.2 Feature-Feature Correlation Matrix")

corr_cols = [
    "Price", "Cylinders", "Is_Luxury", "Car_Age", "Mileage",
    "Mileage_Per_Year", "Condition_Score", "Feature_Count",
]
fig = correlation_heatmap(df_eng, corr_cols, title="Correlation Matrix -- Key Features", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Multicollinearity Observations:**
- **Year and Car_Age** have a perfect negative correlation (-1.0) by construction -- we use only 
  Car_Age in modeling to avoid redundancy
- **Cylinders and Is_Luxury** show moderate correlation (0.48) -- luxury brands tend to have 
  more cylinders, but the relationship is not perfect (Porsche makes 4-cylinder cars too)
- **Mileage and Mileage_Per_Year** have moderate correlation (~0.5) but capture different 
  aspects (total vs intensity)
- No severe multicollinearity among the primary features, so no features need to be dropped 
  for this reason alone
""")

st.markdown("---")

# ============================================================
# Section 6: Final Feature Set
# ============================================================
st.header("6. Final Feature Set for Modeling")

st.markdown("""
The final feature set used across all ML tasks combines the engineered numerical features 
with encoded categorical features. Here is the complete list:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numerical Features (15)**")
    num_feats = get_model_features()
    num_feat_df = pd.DataFrame({
        "Feature": num_feats,
        "Type": [
            "Original", "Original", "Cleaned", "Derived", "Derived",
            "Derived", "Extracted", "Extracted", "Derived",
            "Binary (NLP)", "Binary (NLP)", "Binary (NLP)",
            "Binary (NLP)", "Binary (NLP)", "Binary (NLP)",
        ],
    })
    st.dataframe(num_feat_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Categorical Features (6)**")
    cat_feats = get_categorical_features()
    cat_feat_df = pd.DataFrame({
        "Feature": cat_feats,
        "Encoding": ["Ordinal"] * len(cat_feats),
        "Unique Values": [df_eng[f].nunique() for f in cat_feats],
    })
    st.dataframe(cat_feat_df, use_container_width=True, hide_index=True)

st.markdown("""
**Total input features for regression:** 21 (15 numerical + 6 categorical encoded as ordinal)

**Preprocessing pipeline (applied at training time):**
1. StandardScaler on all numerical features (zero mean, unit variance)
2. OrdinalEncoder on all categorical features (with unknown handling)
3. Applied via scikit-learn ColumnTransformer for consistent, reproducible preprocessing

The preprocessing pipeline is fitted on training data only and saved as a joblib artifact, 
ensuring no data leakage from test or production data.
""")

st.markdown("---")

# ============================================================
# Section 7: Feature Engineering Impact
# ============================================================
st.header("7. Impact of Feature Engineering")

st.markdown("""
To demonstrate the value of feature engineering, consider the predictive information captured 
by each group of features:
""")

impact_data = pd.DataFrame({
    "Feature Group": [
        "Raw Numerical Only (Year, Mileage, Cylinders)",
        "+ Derived Features (Car_Age, Mileage_Per_Year, Log_Mileage)",
        "+ NLP Features (Condition_Score, Feature_Count, binary indicators)",
        "+ Domain Features (Is_Luxury)",
        "+ Categorical (Make, Body Type, Location, etc.)",
    ],
    "Cumulative Features": [3, 6, 15, 16, 22],
    "Information Added": [
        "Basic vehicle specifications",
        "Normalized and transformed metrics",
        "Condition and amenity information from free text",
        "Expert knowledge about market segmentation",
        "Full brand, type, and geographic context",
    ],
})

st.dataframe(impact_data, use_container_width=True, hide_index=True)

st.markdown("""
**Key Takeaway:** Feature engineering transforms a simple 3-number description of a car 
(year, mileage, cylinders) into a rich 22-feature representation that captures brand prestige, 
condition, amenities, market segment, geographic context, and derived metrics. This 
comprehensive representation is what enables models to make accurate predictions. The 
difference between a model trained on raw features vs engineered features is typically 
10-30 percentage points of R-squared improvement.
""")
