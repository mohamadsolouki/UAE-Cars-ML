"""
UAE Used Cars ML -- A Comprehensive Machine Learning Analysis
Main application entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="UAE Used Cars ML",
    page_icon="--",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #e8f4f8;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 3px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("### Navigation")
st.sidebar.markdown(
    "Use the pages in the sidebar to navigate through the analysis. "
    "Each page covers a different stage of the machine learning pipeline."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Stages")
st.sidebar.markdown("""
1. **Data Overview** -- Explore the raw dataset
2. **Exploratory Analysis** -- Visual insights and patterns
3. **Feature Engineering** -- Data transformations
4. **Modeling** -- ML model training and evaluation
5. **Predictions** -- Interactive prediction tools
6. **About** -- Project documentation
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Streamlit, scikit-learn, XGBoost, and Plotly"
)

# --- Main Page ---
st.markdown('<div class="main-header">UAE Used Cars -- Machine Learning Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'A comprehensive end-to-end data science project analyzing 10,000 used car listings '
    'from the UAE market. This application serves as both an analytical tool and '
    'educational material covering the full ML pipeline.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Listings", value="10,000")
with col2:
    st.metric(label="Car Makes", value="65")
with col3:
    st.metric(label="Emirates Covered", value="7")
with col4:
    st.metric(label="Year Range", value="2005-2024")

st.markdown("---")

# Project overview
st.header("Project Overview")

st.markdown("""
This project demonstrates a complete machine learning workflow applied to real-world-style data 
from the UAE used car market. The analysis covers every stage of a data science project, 
from initial data exploration through model deployment.
""")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("What This Project Covers")
    st.markdown("""
    **Data Understanding and Cleaning**
    - Handling mixed data types (Cylinders column contains both numbers and 'Unknown')
    - Standardizing inconsistent text fields (Location whitespace, Make/Model casing)
    - Extracting structured information from free-text descriptions using NLP
    - Identifying and documenting data quality issues

    **Exploratory Data Analysis**
    - Distribution analysis for all numerical and categorical features
    - Price drivers identification through bivariate and multivariate analysis
    - UAE market-specific insights (Dubai vs other emirates, luxury segment)
    - Advanced interactive visualizations with Plotly

    **Feature Engineering**
    - Derived features (Car Age, Mileage per Year, Price per Km)
    - Text feature extraction from descriptions (car features, condition)
    - Categorical encoding strategies for different cardinality levels
    - Feature selection and correlation analysis
    """)

with col_right:
    st.subheader("Machine Learning Tasks")
    st.markdown("""
    **Task 1: Price Prediction (Regression)**
    - Predict the price of a used car based on its attributes
    - Compare 6 models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
    - Best model: Random Forest with cross-validated evaluation

    **Task 2: Body Type Classification**
    - Predict the body type (SUV, Sedan, Coupe, etc.) from car features
    - Random Forest and Gradient Boosting classifiers
    - Confusion matrix analysis and per-class metrics

    **Task 3: Market Segmentation (Clustering)**
    - K-Means clustering to identify natural market segments
    - Elbow method and silhouette analysis for optimal K
    - PCA visualization of cluster structure

    **Task 4: Anomaly Detection**
    - Isolation Forest to identify unusually priced cars
    - Flag overpriced and underpriced listings
    - Deal quality assessment for price negotiation
    """)

st.markdown("---")

# Key findings preview
st.header("Key Findings at a Glance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="insight-box">
        <strong>Market Structure:</strong> The UAE used car market shows a clear 
        two-tier structure -- a mass-market segment (88% of listings) and a 
        luxury segment (12%) with dramatically different price dynamics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Price Drivers:</strong> The number of cylinders (correlation: 0.45) is 
        the strongest single predictor of price, followed by brand prestige and 
        body type. Interestingly, year and mileage show weak linear correlation 
        with price, suggesting non-linear relationships.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-box">
        <strong>Geographic Patterns:</strong> Dubai dominates with 80% of listings. 
        Dubai listings command significantly higher average prices (278K AED) compared to 
        Sharjah (101K AED) and Abu Dhabi (128K AED), driven by a concentration of 
        luxury inventory.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Brand Landscape:</strong> Mercedes-Benz leads with 15% market share, 
        followed by Nissan (9%) and Toyota (9%). Japanese brands dominate the 
        budget segment (60-110K AED) while German and British brands span 
        a much wider price range.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### Getting Started

Navigate through the analysis using the sidebar pages. Each page is designed to be 
self-contained with explanations, visualizations, and interpretations -- structured 
like course material that walks you through the complete data science process.

**Recommended sequence:** Data Overview -> Exploratory Analysis -> Feature Engineering -> Modeling -> Predictions
""")
