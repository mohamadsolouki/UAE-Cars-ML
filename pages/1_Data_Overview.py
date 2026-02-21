"""
Page 1: Data Overview
Provides a comprehensive introduction to the UAE Used Cars dataset,
including structure, types, statistics, and data quality assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_raw_data, load_clean_data

st.set_page_config(page_title="Data Overview | UAE Cars ML", layout="wide", page_icon="--")

st.title("Data Overview")
st.markdown(
    "This page provides a detailed look at the raw dataset before any transformations. "
    "Understanding the structure, types, and quality of data is the essential first step "
    "in any data science project."
)
st.markdown("---")

# Load data
df_raw = load_raw_data()
df_clean = load_clean_data()

# ============================================================
# Section 1: Dataset Summary
# ============================================================
st.header("1. Dataset Summary")

st.markdown("""
The dataset contains **10,000 listings** of used cars from the UAE market. 
Each listing includes 12 attributes describing the car's specifications, condition, 
and market context. The data spans model years from 2005 to 2024 across 65 car manufacturers 
and all 7 UAE emirates.
""")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", f"{len(df_raw):,}")
col2.metric("Columns", f"{len(df_raw.columns)}")
col3.metric("Numerical", "3")
col4.metric("Categorical", "8")
col5.metric("Free Text", "1")

st.markdown("---")

# ============================================================
# Section 2: Column Descriptions
# ============================================================
st.header("2. Column Descriptions")

st.markdown("""
Understanding each column is critical before performing any analysis. Below is a detailed 
breakdown of every attribute in the dataset, including its type, range of values, 
and role in the analysis.
""")

column_info = pd.DataFrame({
    "Column": df_raw.columns,
    "Type": [str(df_raw[c].dtype) for c in df_raw.columns],
    "Non-Null": [df_raw[c].notna().sum() for c in df_raw.columns],
    "Unique Values": [df_raw[c].nunique() for c in df_raw.columns],
    "Sample Values": [
        ", ".join(df_raw[c].astype(str).unique()[:4]) for c in df_raw.columns
    ],
})

column_descriptions = {
    "Make": "Manufacturer / brand of the car (e.g., toyota, bmw, mercedes-benz)",
    "Model": "Specific model name (e.g., camry, x5, g-class)",
    "Year": "Manufacturing year of the vehicle",
    "Price": "Listed price in UAE Dirhams (AED)",
    "Mileage": "Odometer reading in kilometers",
    "Body Type": "Vehicle body classification (SUV, Sedan, Coupe, etc.)",
    "Cylinders": "Number of engine cylinders (contains 'Unknown' values)",
    "Transmission": "Type of transmission (Automatic or Manual)",
    "Fuel Type": "Type of fuel (Gasoline, Diesel, Electric, Hybrid)",
    "Color": "Exterior color of the vehicle",
    "Location": "UAE emirate where the car is listed",
    "Description": "Free-text description with features and condition information",
}
column_info["Description"] = column_info["Column"].map(column_descriptions)

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True,
    height=460,
)

st.markdown("---")

# ============================================================
# Section 3: Raw Data Preview
# ============================================================
st.header("3. Raw Data Preview")

st.markdown("""
Below is an interactive view of the raw dataset. You can sort by any column, 
scroll through records, and get a feel for the data before any cleaning steps.
""")

# Filters
with st.expander("Filter Options", expanded=False):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        selected_makes = st.multiselect(
            "Filter by Make",
            options=sorted(df_raw["Make"].unique()),
            default=[],
        )
    with filter_col2:
        year_range = st.slider(
            "Year Range",
            min_value=int(df_raw["Year"].min()),
            max_value=int(df_raw["Year"].max()),
            value=(int(df_raw["Year"].min()), int(df_raw["Year"].max())),
        )
    with filter_col3:
        price_range = st.slider(
            "Price Range (AED)",
            min_value=int(df_raw["Price"].min()),
            max_value=int(df_raw["Price"].max()),
            value=(int(df_raw["Price"].min()), int(df_raw["Price"].max())),
            step=10000,
        )

filtered = df_raw.copy()
if selected_makes:
    filtered = filtered[filtered["Make"].isin(selected_makes)]
filtered = filtered[
    (filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])
]
filtered = filtered[
    (filtered["Price"] >= price_range[0]) & (filtered["Price"] <= price_range[1])
]

st.markdown(f"Showing **{len(filtered):,}** of {len(df_raw):,} records")
st.dataframe(filtered, use_container_width=True, height=400, hide_index=True)

st.markdown("---")

# ============================================================
# Section 4: Summary Statistics
# ============================================================
st.header("4. Summary Statistics")

st.markdown("""
Statistical summaries help us understand the central tendency, spread, and range of 
numerical features. Key observations are highlighted below the table.
""")

tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])

with tab1:
    num_stats = df_raw.describe().T
    num_stats = num_stats.round(2)
    st.dataframe(num_stats, use_container_width=True)

    st.markdown("""
    **Key Observations:**
    
    - **Price** has extreme right skew: the mean (245,235 AED) is more than double the median 
      (102,766 AED), indicating a long tail of luxury vehicles pulling the average up. 
      The standard deviation (470,977 AED) exceeds the mean, confirming high dispersion.
    - **Mileage** follows a roughly uniform distribution between 10K and 300K km, with a 
      mean around 155K km. This is unusually even and suggests the data may be synthetically generated.
    - **Year** is uniformly distributed from 2005 to 2024  with approximately 500 cars per year, 
      which is another indicator of synthetic generation but provides balanced representation 
      across all model years.
    """)

with tab2:
    cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
    cat_stats = pd.DataFrame({
        "Column": cat_cols,
        "Unique Values": [df_raw[c].nunique() for c in cat_cols],
        "Most Frequent": [df_raw[c].mode().iloc[0] for c in cat_cols],
        "Frequency of Top": [df_raw[c].value_counts().iloc[0] for c in cat_cols],
        "% of Top": [
            f"{df_raw[c].value_counts().iloc[0] / len(df_raw) * 100:.1f}%"
            for c in cat_cols
        ],
    })
    st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key Observations:**
    
    - **Make** has 65 unique manufacturers, with Mercedes-Benz being the most common (14.9%).
    - **Model** is the highest-cardinality feature with 488 unique values -- this poses 
      challenges for encoding in ML models.
    - **Body Type** is dominated by SUVs (46.1%) and Sedans (27.9%), reflecting UAE market preferences.
    - **Transmission** is heavily skewed toward Automatic (96.3%), which is typical for the UAE market.
    - **Fuel Type** is predominantly Gasoline (97.1%), with very few Electric (1.1%) and Hybrid (0.2%) listings.
    """)

st.markdown("---")

# ============================================================
# Section 5: Data Quality Assessment
# ============================================================
st.header("5. Data Quality Assessment")

st.markdown("""
Before proceeding to analysis and modeling, it is essential to identify and document 
all data quality issues. This assessment drives the cleaning strategy in the preprocessing stage.
""")

# Issue 1: Missing values
st.subheader("5.1 Missing Values")
null_counts = df_raw.isnull().sum()
null_pct = (null_counts / len(df_raw) * 100).round(2)
null_df = pd.DataFrame({
    "Column": null_counts.index,
    "Missing Count": null_counts.values,
    "Missing %": null_pct.values,
})
null_df = null_df[null_df["Missing Count"] > 0]

if len(null_df) == 0:
    st.success("No null values detected in any column.")
else:
    st.dataframe(null_df, use_container_width=True, hide_index=True)

st.markdown("""
While there are no traditional null values, the **Cylinders** column contains the 
string "Unknown" for some entries, which effectively represents missing data. 
Additionally, some **Model** values are "other" which serves as a catch-all placeholder.
""")

# Issue 2: Cylinders mixed types
st.subheader("5.2 Cylinders Column -- Mixed Data Types")
cyl_values = df_raw["Cylinders"].value_counts()
st.dataframe(
    pd.DataFrame({"Value": cyl_values.index, "Count": cyl_values.values}),
    use_container_width=True,
    hide_index=True,
    height=320,
)
st.markdown("""
The Cylinders column is stored as text (object type) because it contains both numeric values 
(3, 4, 5, 6, 8, 10, 12) and the string "Unknown". This requires special handling:

- **Strategy:** Convert to numeric, coerce "Unknown" to NaN, then impute based on the 
  median cylinder count for each Make (brand). This is reasonable because engine size 
  is strongly associated with brand positioning.
""")

# Issue 3: Location whitespace
st.subheader("5.3 Location -- Inconsistent Whitespace")
loc_raw = df_raw["Location"].value_counts()
st.markdown(
    f"The raw Location column has **{df_raw['Location'].nunique()}** unique values "
    f"instead of the expected 7-8 emirates. This is caused by inconsistent leading spaces."
)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**Before Cleaning (Raw)**")
    st.dataframe(
        pd.DataFrame({"Location": loc_raw.index, "Count": loc_raw.values}),
        hide_index=True,
        height=300,
    )
with col_r:
    loc_clean = df_raw["Location"].str.strip().value_counts()
    st.markdown("**After Cleaning (Stripped)**")
    st.dataframe(
        pd.DataFrame({"Location": loc_clean.index, "Count": loc_clean.values}),
        hide_index=True,
        height=300,
    )

st.markdown("""
**Strategy:** Apply `.str.strip()` to remove leading and trailing whitespace. 
After cleaning, we correctly identify 8 emirates: Dubai, Sharjah, Abu Dhabi, 
Ajman, Al Ain, Ras Al Khaimah, Umm Al Qawain, and Fujeirah.
""")

# Issue 4: Description parsing
st.subheader("5.4 Description Column -- Structured Text")

st.markdown("""
The Description column contains semi-structured text following a consistent template:

> "{year} {make} {model} with {feature1}, {feature2}, ... . Condition: {condition}."

This template allows reliable extraction of two valuable pieces of information:
""")

sample_desc = df_raw["Description"].iloc[0]
st.code(sample_desc, language=None)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Extracted Features (6 unique):**")
    for feat in ["Sunroof", "Leather seats", "Navigation system",
                 "Bluetooth", "Rear camera", "Adaptive cruise control"]:
        count = df_clean["Features_List"].apply(lambda x: feat in x).sum()
        st.markdown(f"- {feat}: present in {count:,} listings ({count/len(df_clean)*100:.1f}%)")

with col_b:
    st.markdown("**Extracted Conditions (6 unique):**")
    cond_counts = df_clean["Condition"].value_counts()
    for cond, count in cond_counts.items():
        st.markdown(f"- {cond}: {count:,} listings ({count/len(df_clean)*100:.1f}%)")

st.markdown("---")

# ============================================================
# Section 6: Data Types Summary
# ============================================================
st.header("6. Data Types After Cleaning")

st.markdown("""
After applying all cleaning steps, the dataset is transformed from 12 raw columns to a 
richer representation with parsed features, standardized types, and display-friendly formatting.
The table below shows the cleaned data structure.
""")

clean_preview = df_clean[
    ["Make", "Make_Display", "Model", "Model_Display", "Year", "Price",
     "Mileage", "Body Type", "Cylinders", "Transmission", "Fuel Type",
     "Color", "Location", "Condition"]
].head(10)
st.dataframe(clean_preview, use_container_width=True, hide_index=True)

st.markdown("""
**Cleaning Summary:**
- Location whitespace stripped (16 raw values consolidated to 8 emirates)
- Cylinders converted from mixed text to integer (105 "Unknown" values imputed by Make median)
- Make and Model display versions created with proper capitalization
- Condition extracted from Description as a separate categorical column
- Car features extracted from Description as a binary feature list
""")
