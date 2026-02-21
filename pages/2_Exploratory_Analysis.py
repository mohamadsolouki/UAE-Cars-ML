"""
Page 2: Exploratory Data Analysis
Comprehensive visual exploration of the UAE Used Cars dataset with
univariate, bivariate, and multivariate analysis plus market insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_clean_data, load_engineered_data
from src.visualization import (
    histogram, box_plot, violin_plot, bar_chart, scatter_plot,
    correlation_heatmap, pie_chart, sunburst_chart, line_chart,
    apply_layout, COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT,
)

st.set_page_config(page_title="Exploratory Analysis | UAE Cars ML", layout="wide", page_icon="--")

st.title("Exploratory Data Analysis")
st.markdown(
    "EDA is the process of visually and statistically summarizing the main characteristics "
    "of a dataset. This page presents a comprehensive exploration organized into univariate, "
    "bivariate, and multivariate analyses, followed by UAE market-specific insights."
)
st.markdown("---")

df = load_clean_data()
df_eng = load_engineered_data()

# ============================================================
# Section 1: Univariate Analysis
# ============================================================
st.header("1. Univariate Analysis")
st.markdown(
    "Univariate analysis examines each variable independently to understand its "
    "distribution, central tendency, and spread. This reveals the basic structure of the data."
)

# --- Price Distribution ---
st.subheader("1.1 Price Distribution")

col1, col2 = st.columns(2)
with col1:
    fig = histogram(df, "Price", title="Price Distribution (AED)", nbins=80)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = histogram(df_eng, "Log_Price", title="Log-Transformed Price Distribution", nbins=60, color=COLOR_SECONDARY)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** The price distribution exhibits extreme right skew, with most cars 
priced below 300,000 AED but a long tail extending to nearly 15 million AED (McLaren P1). 
The log transformation (right panel) reveals a more symmetric distribution, suggesting 
that log-price is closer to normal and may be better suited as a regression target.

Key statistics:
- **Median price:** 102,766 AED -- this is a more representative "typical" price than the mean
- **Mean price:** 245,235 AED -- inflated by luxury vehicles
- **IQR:** 50,353 to 231,248 AED -- where most of the market sits
- **Price ratio (max/min):** ~2,043x -- enormous range from budget to ultra-luxury
""")

# --- Year, Mileage distributions ---
st.subheader("1.2 Year and Mileage Distributions")

col1, col2 = st.columns(2)
with col1:
    year_counts = df["Year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Year", "Count"]
    fig = px.bar(year_counts, x="Year", y="Count", color_discrete_sequence=[COLOR_PRIMARY])
    fig = apply_layout(fig, title="Distribution of Model Years", height=400)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = histogram(df, "Mileage", title="Mileage Distribution (km)", nbins=60, color=COLOR_ACCENT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** 
- **Year** is approximately uniformly distributed from 2005 to 2024 with ~500 cars per year. 
  This even spread means each model year is equally represented, which is unusual for real-world 
  data (where newer years typically have more listings) and suggests synthetic generation.
- **Mileage** also follows a near-uniform distribution from 10K to 300K km. Real used car 
  mileage typically follows a right-skewed distribution concentrated around 50K-150K km.
""")

# --- Categorical distributions ---
st.subheader("1.3 Categorical Feature Distributions")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Make (Brand)", "Body Type", "Fuel Type", "Transmission", "Color", "Location"
])

with tab1:
    make_counts = df["Make_Display"].value_counts().head(20).reset_index()
    make_counts.columns = ["Make", "Count"]
    fig = px.bar(make_counts, x="Count", y="Make", orientation="h",
                 color_discrete_sequence=[COLOR_PRIMARY])
    fig = apply_layout(fig, "Top 20 Car Makes by Listing Count", height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Mercedes-Benz leads the market with 1,486 listings (14.9%), followed by Nissan (925) 
    and Toyota (893). The top 3 brands alone account for 33% of all listings. 
    The presence of luxury brands like Porsche (#7), Rolls-Royce (#20), and Land Rover (#6) 
    in the top 20 reflects the UAE's affluent car market.
    """)

with tab2:
    bt_counts = df["Body Type"].value_counts().reset_index()
    bt_counts.columns = ["Body Type", "Count"]
    fig = px.bar(bt_counts, x="Count", y="Body Type", orientation="h",
                 color_discrete_sequence=[COLOR_SECONDARY])
    fig = apply_layout(fig, "Body Type Distribution", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    SUVs dominate with 46.1% of all listings, followed by Sedans at 27.9%. Together they 
    represent nearly 75% of the market. This aligns with the UAE's preference for larger 
    vehicles suited to the terrain and climate. Sports cars, convertibles, and specialty 
    vehicles make up the remaining 25%.
    """)

with tab3:
    ft_counts = df["Fuel Type"].value_counts().reset_index()
    ft_counts.columns = ["Fuel Type", "Count"]
    fig = pie_chart(ft_counts["Count"], ft_counts["Fuel Type"],
                    title="Fuel Type Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Gasoline dominates overwhelmingly at 97.1% of listings. Diesel (1.5%), Electric (1.1%), 
    and Hybrid (0.2%) represent a tiny fraction. This is consistent with the UAE market where 
    fuel subsidies traditionally favor gasoline vehicles, though electric adoption is growing.
    """)

with tab4:
    trans_counts = df["Transmission"].value_counts().reset_index()
    trans_counts.columns = ["Transmission", "Count"]
    fig = pie_chart(trans_counts["Count"], trans_counts["Transmission"],
                    title="Transmission Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Automatic transmission accounts for 96.3% of listings, with only 3.7% manual. 
    The UAE market strongly favors automatic vehicles due to urban driving conditions 
    and heavy traffic in cities like Dubai and Abu Dhabi.
    """)

with tab5:
    color_counts = df["Color"].value_counts().reset_index()
    color_counts.columns = ["Color", "Count"]
    fig = px.bar(color_counts, x="Color", y="Count", color_discrete_sequence=[COLOR_PRIMARY])
    fig.update_layout(xaxis_tickangle=45)
    fig = apply_layout(fig, "Color Distribution", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    White (33.6%), Black (21.3%), and Grey (13.1%) are the three most popular colors, 
    together making up 68% of all listings. White is the most practical color for the UAE's 
    extreme heat as it reflects sunlight. Specialty colors like Teal, Tan, and Purple are rare.
    """)

with tab6:
    loc_counts = df["Location"].value_counts().reset_index()
    loc_counts.columns = ["Location", "Count"]
    fig = px.bar(loc_counts, x="Location", y="Count", color_discrete_sequence=[COLOR_ACCENT])
    fig = apply_layout(fig, "Listings by Emirate", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Dubai dominates with 80.1% of all listings, followed by Sharjah (10.0%) and Abu Dhabi (7.3%). 
    The remaining emirates (Ajman, Al Ain, Ras Al Khaimah, Umm Al Qawain, Fujeirah) together 
    account for less than 3% of listings. This concentration reflects Dubai's position as the 
    primary commercial hub and used car trading center in the UAE.
    """)

st.markdown("---")

# ============================================================
# Section 1.4: Outlier Analysis
# ============================================================
st.subheader("1.4 Price Outlier Analysis")
st.markdown("""
Given the extreme skewness of the price distribution (skewness = 9.28), outlier analysis is 
critical for understanding data quality and model training decisions. We use the **Interquartile 
Range (IQR) method** to identify statistical outliers.
""")

# Calculate outlier statistics
Q1_price = df["Price"].quantile(0.25)
Q3_price = df["Price"].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound = max(0, Q1_price - 1.5 * IQR_price)
upper_bound = Q3_price + 1.5 * IQR_price

outliers_df = df[(df["Price"] < lower_bound) | (df["Price"] > upper_bound)]
upper_outliers = df[df["Price"] > upper_bound]
n_outliers = len(outliers_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Q1 (25th percentile)", f"{Q1_price:,.0f} AED")
col2.metric("Q3 (75th percentile)", f"{Q3_price:,.0f} AED")
col3.metric("IQR", f"{IQR_price:,.0f} AED")
col4.metric("Outlier Threshold", f"{upper_bound:,.0f} AED")

col1, col2, col3 = st.columns(3)
col1.metric("Total Outliers", f"{n_outliers:,}", delta=f"{n_outliers/len(df)*100:.1f}% of data")
col2.metric("Upper Outliers", f"{len(upper_outliers):,}", delta="High-priced vehicles")
col3.metric("Outlier Price Range", f"{upper_outliers['Price'].min():,.0f} - {upper_outliers['Price'].max():,.0f} AED")

# Box plot with outliers highlighted
fig = go.Figure()
fig.add_trace(go.Box(
    x=df["Price"],
    name="All Cars",
    marker_color=COLOR_PRIMARY,
    boxpoints="outliers",
))
fig.update_layout(xaxis_title="Price (AED)")
fig = apply_layout(fig, "Price Distribution with Outliers (Box Plot)", height=300)
st.plotly_chart(fig, use_container_width=True)

# Outlier composition
st.markdown("**Outlier Composition by Brand:**")
outlier_makes = upper_outliers["Make_Display"].value_counts().head(10).reset_index()
outlier_makes.columns = ["Make", "Outlier Count"]

col1, col2 = st.columns([1, 1])
with col1:
    fig = px.bar(outlier_makes, x="Outlier Count", y="Make", orientation="h",
                 color_discrete_sequence=[COLOR_WARN])
    fig = apply_layout(fig, "Top 10 Brands in Outlier Range", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    outlier_bt = upper_outliers["Body Type"].value_counts().reset_index()
    outlier_bt.columns = ["Body Type", "Count"]
    fig = pie_chart(outlier_bt["Count"], outlier_bt["Body Type"],
                    title="Outlier Distribution by Body Type")
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Key Findings:**

1. **{n_outliers:,} listings ({n_outliers/len(df)*100:.1f}%)** are classified as price outliers
2. Upper outliers (>${upper_bound:,.0f} AED) are dominated by:
   - **Luxury brands:** McLaren, Rolls-Royce, Ferrari, Lamborghini, Bentley
   - **Premium variants:** High-end Mercedes-Benz (AMG, Maybach), Porsche
3. **Sports Cars and SUVs** comprise the majority of outlier body types
4. These outliers represent the **luxury segment** which behaves differently from mass market

**Modeling Implications:**
- Training on raw prices leads to poor R² due to extreme variance
- **Log-transformation** of price helps normalize the distribution (skewness drops from 9.28 to ~0.8)
- Separate models for luxury vs mass-market may improve predictions
- Or using quantile regression for better handling of price extremes
""")

st.markdown("---")

# ============================================================
# Section 1.5: Luxury vs Mass Market Deep Dive
# ============================================================
st.subheader("1.5 Luxury vs Mass-Market Deep Dive")
st.markdown("""
The UAE used car market exhibits a distinct **two-tier structure**. This section provides 
a comprehensive comparison between luxury and mass-market segments.
""")

# Define segments
luxury_makes = ["ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
                "aston-martin", "mclaren", "maybach", "porsche", "bugatti"]
df_lux = df[df["Make"].isin(luxury_makes)].copy()
df_mass = df[~df["Make"].isin(luxury_makes)].copy()

# Key metrics comparison
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Luxury Segment")
    st.metric("Total Listings", f"{len(df_lux):,}", delta=f"{len(df_lux)/len(df)*100:.1f}% of market")
    st.metric("Average Price", f"{df_lux['Price'].mean():,.0f} AED")
    st.metric("Median Price", f"{df_lux['Price'].median():,.0f} AED")
    st.metric("Avg Cylinders", f"{df_lux['Cylinders'].mean():.1f}")
    st.metric("Avg Year", f"{df_lux['Year'].mean():.0f}")

with col2:
    st.markdown("### Mass Market")
    st.metric("Total Listings", f"{len(df_mass):,}", delta=f"{len(df_mass)/len(df)*100:.1f}% of market")
    st.metric("Average Price", f"{df_mass['Price'].mean():,.0f} AED")
    st.metric("Median Price", f"{df_mass['Price'].median():,.0f} AED")
    st.metric("Avg Cylinders", f"{df_mass['Cylinders'].mean():.1f}")
    st.metric("Avg Year", f"{df_mass['Year'].mean():.0f}")

# Price comparison violin
df_segment = df.copy()
df_segment["Segment"] = df_segment["Make"].apply(lambda x: "Luxury" if x in luxury_makes else "Mass Market")

fig = px.violin(df_segment, x="Segment", y="Price", color="Segment", box=True,
                color_discrete_sequence=[COLOR_WARN, COLOR_PRIMARY])
fig = apply_layout(fig, "Price Distribution by Market Segment", height=450)
st.plotly_chart(fig, use_container_width=True)

# Feature differences
st.markdown("**Feature Availability by Segment:**")
feature_cols = ["has_sunroof", "has_leather_seats", "has_navigation_system", 
                "has_bluetooth", "has_rear_camera", "has_adaptive_cruise_control"]

if all(col in df_eng.columns for col in feature_cols):
    df_eng_seg = df_eng.copy()
    df_eng_seg["Segment"] = df_eng_seg["Make"].apply(lambda x: "Luxury" if x in luxury_makes else "Mass Market")
    
    feature_summary = []
    for feat in feature_cols:
        feat_name = feat.replace("has_", "").replace("_", " ").title()
        lux_pct = df_eng_seg[df_eng_seg["Segment"] == "Luxury"][feat].mean() * 100
        mass_pct = df_eng_seg[df_eng_seg["Segment"] == "Mass Market"][feat].mean() * 100
        feature_summary.append({
            "Feature": feat_name,
            "Luxury (%)": round(lux_pct, 1),
            "Mass Market (%)": round(mass_pct, 1),
            "Difference": round(lux_pct - mass_pct, 1),
        })
    
    feat_df = pd.DataFrame(feature_summary)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Luxury",
        x=feat_df["Feature"],
        y=feat_df["Luxury (%)"],
        marker_color=COLOR_WARN,
    ))
    fig.add_trace(go.Bar(
        name="Mass Market",
        x=feat_df["Feature"],
        y=feat_df["Mass Market (%)"],
        marker_color=COLOR_PRIMARY,
    ))
    fig.update_layout(barmode="group", xaxis_tickangle=45)
    fig = apply_layout(fig, "Feature Penetration Rate by Segment (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Cylinder distribution comparison
st.markdown("**Cylinder Distribution by Segment:**")
col1, col2 = st.columns(2)

with col1:
    lux_cyl = df_lux["Cylinders"].value_counts().sort_index().reset_index()
    lux_cyl.columns = ["Cylinders", "Count"]
    fig = px.bar(lux_cyl, x="Cylinders", y="Count", color_discrete_sequence=[COLOR_WARN])
    fig = apply_layout(fig, "Luxury Segment Cylinders", height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    mass_cyl = df_mass["Cylinders"].value_counts().sort_index().reset_index()
    mass_cyl.columns = ["Cylinders", "Count"]
    fig = px.bar(mass_cyl, x="Cylinders", y="Count", color_discrete_sequence=[COLOR_PRIMARY])
    fig = apply_layout(fig, "Mass Market Cylinders", height=350)
    st.plotly_chart(fig, use_container_width=True)

# Body type comparison
st.markdown("**Body Type Preferences by Segment:**")
lux_bt = df_lux["Body Type"].value_counts().head(5).reset_index()
lux_bt.columns = ["Body Type", "Count"]
lux_bt["Segment"] = "Luxury"

mass_bt = df_mass["Body Type"].value_counts().head(5).reset_index()
mass_bt.columns = ["Body Type", "Count"]
mass_bt["Segment"] = "Mass Market"

combined_bt = pd.concat([lux_bt, mass_bt])
fig = px.bar(combined_bt, x="Body Type", y="Count", color="Segment",
             barmode="group", color_discrete_sequence=[COLOR_WARN, COLOR_PRIMARY])
fig = apply_layout(fig, "Top 5 Body Types by Market Segment", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Key Segment Differences:**

| Metric | Luxury | Mass Market | Ratio |
|--------|--------|-------------|-------|
| Average Price | {df_lux['Price'].mean():,.0f} AED | {df_mass['Price'].mean():,.0f} AED | **{df_lux['Price'].mean()/df_mass['Price'].mean():.1f}x** |
| Median Price | {df_lux['Price'].median():,.0f} AED | {df_mass['Price'].median():,.0f} AED | **{df_lux['Price'].median()/df_mass['Price'].median():.1f}x** |
| Avg Cylinders | {df_lux['Cylinders'].mean():.1f} | {df_mass['Cylinders'].mean():.1f} | **{df_lux['Cylinders'].mean()/df_mass['Cylinders'].mean():.1f}x** |
| Market Share | {len(df_lux)/len(df)*100:.1f}% | {len(df_mass)/len(df)*100:.1f}% | 1:{len(df_mass)//len(df_lux)} |

**Business Insights:**
1. **Luxury brands command 5-6x higher prices** on average
2. **Cylinders correlate strongly with segment** - luxury averages ~8 cylinders vs ~5.5 for mass market
3. **Sports Cars dominate luxury listings** while SUVs lead mass market
4. **Feature penetration is similar** across segments, suggesting features don't differentiate price tiers
5. The **two-tier market structure** explains why linear models struggle - different pricing mechanisms apply
""")

st.markdown("---")

# ============================================================
# Section 2: Bivariate Analysis
# ============================================================
st.header("2. Bivariate Analysis")
st.markdown(
    "Bivariate analysis examines relationships between pairs of variables, "
    "revealing correlations, group differences, and predictive patterns."
)

# --- Price vs Year ---
st.subheader("2.1 Price vs Year (Depreciation Analysis)")

fig = scatter_plot(df, "Year", "Price", title="Price vs Model Year", trendline="lowess")
st.plotly_chart(fig, use_container_width=True)

# Depreciation by brand
st.markdown("**Depreciation Curves by Brand (Top 6)**")
top_makes = df["Make"].value_counts().head(6).index.tolist()
dep_data = df[df["Make"].isin(top_makes)].groupby(["Year", "Make_Display"])["Price"].median().reset_index()
fig = px.line(dep_data, x="Year", y="Price", color="Make_Display", markers=True,
              color_discrete_sequence=COLORS)
fig = apply_layout(fig, "Median Price by Year -- Top 6 Brands", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** The scatter plot shows a weak overall relationship between year and price 
(correlation near zero). This is because the dataset mixes economy and luxury segments. 
However, the brand-specific depreciation curves reveal important patterns:

- **Mercedes-Benz** shows the strongest price range across years, with newer models 
  maintaining higher values
- **Nissan and Toyota** show relatively flat depreciation curves, consistent with their 
  reputation for value retention
- The weak overall correlation (0.0002) is a classic example of **Simpson's Paradox** -- the 
  relationship appears when stratified by brand but vanishes in aggregate
""")

# --- Price vs Mileage ---
st.subheader("2.2 Price vs Mileage")

fig = scatter_plot(df, "Mileage", "Price", title="Price vs Mileage", trendline="lowess")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** Similar to year, the overall mileage-price correlation is near zero (0.01). 
The LOWESS trendline shows an essentially flat relationship across all mileage levels. This 
counterintuitive finding occurs because mileage alone does not determine price -- brand prestige 
and vehicle type are far more influential. A low-mileage Nissan Altima will still be priced 
below a high-mileage Rolls-Royce.
""")

# --- Price by Make (Box Plots) ---
st.subheader("2.3 Price Distribution by Make")
top_15_makes = df["Make_Display"].value_counts().head(15).index.tolist()
df_top15 = df[df["Make_Display"].isin(top_15_makes)]

# Sort by median price
make_order = df_top15.groupby("Make_Display")["Price"].median().sort_values(ascending=False).index.tolist()
fig = px.box(df_top15, x="Make_Display", y="Price",
             category_orders={"Make_Display": make_order},
             color_discrete_sequence=[COLOR_PRIMARY])
fig.update_layout(xaxis_tickangle=45)
fig = apply_layout(fig, "Price Distribution by Make (Top 15 Brands)", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** The box plots reveal striking differences in price distributions across brands:

- **Porsche and Land Rover** show the highest median prices among volume brands (~197K and ~231K AED), 
  with significant upper outliers
- **Mercedes-Benz** has an enormous range from ~14K to ~4.8M AED, reflecting its diverse lineup from 
  entry-level to ultra-luxury (AMG, Maybach-class)
- **Mitsubishi and Hyundai** cluster tightly below 100K AED with small IQR, indicating price consistency
- The outlier pattern is consistent across brands -- every brand has some premium models pulling 
  the distribution upward
""")

# --- Price by Body Type (Violin) ---
st.subheader("2.4 Price by Body Type")

# Filter to major body types for readability
major_bt = df["Body Type"].value_counts().head(8).index.tolist()
df_bt = df[df["Body Type"].isin(major_bt)]
bt_order = df_bt.groupby("Body Type")["Price"].median().sort_values(ascending=False).index.tolist()

fig = px.violin(df_bt, x="Body Type", y="Price", box=True,
                category_orders={"Body Type": bt_order},
                color_discrete_sequence=[COLOR_SECONDARY])
fig.update_layout(xaxis_tickangle=45)
fig = apply_layout(fig, "Price Distribution by Body Type", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** Body type strongly segments the market:

- **Sports Cars** command the highest prices (median ~249K, mean ~706K AED) with heavy right skew 
  from supercars like McLaren and Ferrari
- **SUVs**, despite being the most common, have a wide price range reflecting models from 
  budget (Hyundai Tucson) to luxury (Range Rover, G-Class)
- **Hatchbacks** are consistently the most affordable body type (median ~51K AED)
- The violin shapes show that most body types have bimodal or skewed distributions, 
  reinforcing the two-tier market structure
""")

# --- Price by Location ---
st.subheader("2.5 Price by Location (Emirates)")

main_locs = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
df_loc = df[df["Location"].isin(main_locs)]
fig = px.box(df_loc, x="Location", y="Price", color="Location",
             color_discrete_sequence=COLORS)
fig = apply_layout(fig, "Price Distribution by Emirate", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** Dubai shows dramatically higher prices (median ~114K, mean ~278K AED) 
compared to other emirates:

- **Dubai** concentrates luxury inventory, driving up both mean and the range of outliers
- **Abu Dhabi** (median ~84K AED) and **Sharjah** (median ~64K AED) serve the mid-market segment
- **Ajman** (median ~75K AED) has a tighter distribution focused on budget to mid-range vehicles
- The 2.7x median price difference between Dubai and Sharjah represents an opportunity for 
  cross-emirate price arbitrage
""")

# --- Cylinders vs Price ---
st.subheader("2.6 Cylinders vs Price")

df_cyl = df[df["Cylinders"].notna()].copy()
df_cyl["Cylinders_str"] = df_cyl["Cylinders"].astype(int).astype(str) + " cyl"
cyl_order = [f"{c} cyl" for c in sorted(df_cyl["Cylinders"].unique())]
fig = px.box(df_cyl, x="Cylinders_str", y="Price",
             category_orders={"Cylinders_str": cyl_order},
             color_discrete_sequence=[COLOR_ACCENT])
fig = apply_layout(fig, "Price by Number of Cylinders", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** Cylinder count is the strongest single predictor of price (correlation: 0.45):

- **12-cylinder** vehicles have a median price of ~1.15M AED -- these are exclusively 
  ultra-luxury brands (Rolls-Royce, Ferrari, Lamborghini, Bentley)
- **10-cylinder** vehicles (median ~828K AED) are primarily Lamborghini and Audi models
- **8-cylinder** (median ~168K AED) spans a wide range from V8 muscle cars to luxury SUVs
- **4-cylinder** (median ~62K AED) represents the economy segment
- There is a clear near-exponential relationship between cylinders and price
""")

st.markdown("---")

# ============================================================
# Section 3: Multivariate Analysis
# ============================================================
st.header("3. Multivariate Analysis")
st.markdown(
    "Multivariate analysis examines interactions among three or more variables simultaneously, "
    "revealing complex patterns that simpler analyses might miss."
)

# --- Correlation Heatmap ---
st.subheader("3.1 Correlation Matrix")

corr_cols = ["Price", "Year", "Mileage", "Cylinders", "Car_Age",
             "Mileage_Per_Year", "Condition_Score", "Feature_Count",
             "Is_Luxury", "Log_Price"]
fig = correlation_heatmap(df_eng, corr_cols, title="Correlation Matrix of Numerical Features",
                          height=550)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Key Correlations:**

| Feature Pair | Correlation | Interpretation |
|---|---|---|
| Cylinders - Price | 0.45 | Engine size strongly predicts price |
| Is_Luxury - Price | 0.39 | Luxury brand flag captures brand premium |
| Log_Price - Cylinders | 0.54 | Even stronger on log scale |
| Year - Car_Age | -1.00 | Perfect inverse (by construction) |
| Year - Price | ~0.00 | No linear relationship (Simpson's Paradox) |
| Mileage - Price | ~0.01 | No linear relationship |

The near-zero correlations of Year and Mileage with Price highlight why linear models struggle 
with this dataset. The true price relationships are non-linear and heavily mediated by 
categorical variables (brand, body type).
""")

# --- Sunburst Chart ---
st.subheader("3.2 Market Composition")

# Prepare data for sunburst
sunburst_df = df[df["Location"].isin(["Dubai", "Abu Dhabi", "Sharjah"])].copy()
sunburst_agg = (sunburst_df.groupby(["Location", "Body Type", "Fuel Type"])
                .size().reset_index(name="Count"))
fig = sunburst_chart(
    sunburst_agg,
    path=["Location", "Body Type", "Fuel Type"],
    values="Count",
    title="Market Composition: Location > Body Type > Fuel Type",
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** The sunburst chart reveals the hierarchical structure of the UAE car market:

- Dubai's dominance is clear at the outer ring, with SUVs and Sedans forming its largest segments
- Gasoline overwhelmingly dominates across all locations and body types
- Abu Dhabi has a relatively higher proportion of SUVs compared to Sedans
- Electric vehicles are almost exclusively found in Dubai listings
""")

# --- Price by Make and Body Type Heatmap ---
st.subheader("3.3 Average Price Heatmap: Make vs Body Type")

top10_makes = df["Make_Display"].value_counts().head(10).index.tolist()
top6_bt = df["Body Type"].value_counts().head(6).index.tolist()
heatmap_df = df[df["Make_Display"].isin(top10_makes) & df["Body Type"].isin(top6_bt)]
pivot = heatmap_df.pivot_table(values="Price", index="Make_Display", columns="Body Type",
                                aggfunc="median")
pivot = pivot.reindex(index=top10_makes, columns=top6_bt)

fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale="Viridis",
    text=[[f"{v:,.0f}" if pd.notna(v) else "N/A" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10),
    hovertemplate="Make: %{y}<br>Body Type: %{x}<br>Median Price: %{z:,.0f} AED<extra></extra>",
))
fig = apply_layout(fig, "Median Price by Make and Body Type (AED)", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:** This heatmap reveals which Make + Body Type combinations command the 
highest prices. Porsche Coupes, Land Rover SUVs, and BMW Sedans stand out. 
Empty cells (N/A) indicate combinations not present in the data -- for example, 
some brands do not produce certain body types.
""")

st.markdown("---")

# ============================================================
# Section 4: UAE Market Insights
# ============================================================
st.header("4. UAE Market Insights")
st.markdown(
    "This section synthesizes the individual analyses into actionable insights "
    "specific to the UAE used car market."
)

# --- Luxury vs Mass Market ---
st.subheader("4.1 Luxury vs Mass-Market Segments")

luxury_makes = ["ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
                "aston-martin", "mclaren", "maybach", "porsche"]
df_seg = df.copy()
df_seg["Segment"] = df_seg["Make"].apply(lambda x: "Luxury" if x in luxury_makes else "Mass Market")

seg_stats = df_seg.groupby("Segment").agg(
    Count=("Price", "count"),
    Mean_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Avg_Mileage=("Mileage", "mean"),
    Avg_Year=("Year", "mean"),
).reset_index()
seg_stats["Mean_Price"] = seg_stats["Mean_Price"].apply(lambda x: f"{x:,.0f}")
seg_stats["Median_Price"] = seg_stats["Median_Price"].apply(lambda x: f"{x:,.0f}")
seg_stats["Avg_Mileage"] = seg_stats["Avg_Mileage"].apply(lambda x: f"{x:,.0f}")
seg_stats["Avg_Year"] = seg_stats["Avg_Year"].apply(lambda x: f"{x:.1f}")

col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(seg_stats, use_container_width=True, hide_index=True)
with col2:
    seg_counts = df_seg["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig = pie_chart(seg_counts["Count"], seg_counts["Segment"],
                    title="Market Segment Split")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
The UAE used car market has a pronounced two-tier structure:

- **Mass Market (88%)**: Average price ~158K AED. Dominated by Japanese brands (Toyota, Nissan), 
  Korean brands (Hyundai, Kia), and American brands (Ford, Jeep, Chevrolet). These cars are 
  primarily 4-6 cylinder vehicles used for daily transportation.
- **Luxury Segment (12%)**: Average price ~881K AED (5.6x higher). Dominated by European 
  marques. Features predominantly 8-12 cylinder engines with emphasis on SUVs and sports cars.
""")

# --- Color Preferences by Location ---
st.subheader("4.2 Color Preferences by Emirate")

top3_locs = ["Dubai", "Abu Dhabi", "Sharjah"]
top5_colors = df["Color"].value_counts().head(5).index.tolist()
color_loc = df[df["Location"].isin(top3_locs) & df["Color"].isin(top5_colors)]
color_pct = (color_loc.groupby(["Location", "Color"]).size()
             .div(color_loc.groupby("Location").size(), level="Location")
             .mul(100).reset_index(name="Percentage"))

fig = px.bar(color_pct, x="Location", y="Percentage", color="Color",
             barmode="group", color_discrete_map={
                 "White": "#E8E8E8", "Black": "#333333", "Grey": "#999999",
                 "Silver": "#C0C0C0", "Blue": "#4169E1",
             })
fig = apply_layout(fig, "Color Distribution by Emirate (%)", height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
White is universally the most popular color across all emirates, likely driven by the practical 
benefit of heat reflection in the UAE's hot climate. Dubai shows a slightly higher proportion 
of black vehicles, which aligns with the luxury preference in the emirate where aesthetics 
may outweigh thermal practicality.
""")

# --- Top 10 Most Expensive Cars ---
st.subheader("4.3 Most Expensive Listings")

top10 = df.nlargest(10, "Price")[
    ["Make_Display", "Model_Display", "Year", "Price", "Mileage",
     "Body Type", "Cylinders", "Location"]
].copy()
top10["Price"] = top10["Price"].apply(lambda x: f"{x:,.0f} AED")
top10["Mileage"] = top10["Mileage"].apply(lambda x: f"{x:,.0f} km")
top10.index = range(1, 11)
top10.index.name = "Rank"

st.dataframe(top10, use_container_width=True)

st.markdown("""
The most expensive listings are dominated by **McLaren** (P1, Elva, Senna) and **Ferrari** (599), 
all of which are limited-production supercars. The McLaren P1 at nearly 14.7M AED 
tops the list. All top 10 cars are listed in Dubai, confirming the emirate's role as the 
primary luxury car market.

Note: Some year/model combinations (e.g., 2005 McLaren P1, which actually debuted in 2013) 
are historically impossible, confirming this is a synthetically generated dataset.
""")

# --- Condition Analysis ---
st.subheader("4.4 Price by Vehicle Condition")

cond_order = ["No damage", "Minor scratches", "Repainted bumper",
              "Dented door", "Engine repaired", "Accident history"]
df_cond = df.copy()
df_cond["Condition"] = pd.Categorical(df_cond["Condition"], categories=cond_order, ordered=True)

fig = px.box(df_cond, x="Condition", y="Price",
             category_orders={"Condition": cond_order},
             color_discrete_sequence=[COLOR_PRIMARY])
fig = apply_layout(fig, "Price Distribution by Vehicle Condition", height=450)
st.plotly_chart(fig, use_container_width=True)

cond_medians = df.groupby("Condition")["Price"].median().reindex(cond_order)
st.markdown(f"""
**Interpretation:** Surprisingly, the condition does not significantly differentiate prices in 
this dataset. Median prices are relatively similar across all conditions:

- No damage: {cond_medians.iloc[0]:,.0f} AED
- Minor scratches: {cond_medians.iloc[1]:,.0f} AED
- Accident history: {cond_medians.iloc[5]:,.0f} AED

This suggests that condition was randomly assigned during data generation and does not 
reflect real-world pricing dynamics (where accident history would typically reduce value 
by 10-30%). This is an important finding to account for in modeling.
""")

st.markdown("---")

st.markdown("""
### Summary of EDA Findings

1. **Price is heavily right-skewed** with a 2,043x range between cheapest and most expensive
2. **Cylinders (0.45) and luxury brand status (0.39)** are the strongest linear predictors of price
3. **Year and mileage have near-zero correlation** with price in aggregate -- a Simpson's Paradox effect
4. **Dubai dominates** with 80% of listings and significantly higher prices
5. **SUVs and Sedans** together represent 74% of the market
6. **White, Black, and Grey** account for 68% of all vehicle colors
7. **The market has a clear two-tier structure**: mass-market vs luxury, with 5.6x median price difference
8. **Condition appears randomly assigned** and does not meaningfully impact price

These findings inform the feature engineering and modeling strategies on the following pages.
""")
