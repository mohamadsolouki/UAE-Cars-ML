"""
Data loading utilities for the UAE Used Cars dataset.
Handles CSV loading with Streamlit caching for performance.
"""

import pandas as pd
import streamlit as st
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uae_used_cars_10k.csv")


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV file and return as a DataFrame."""
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    """Load and apply basic cleaning to the dataset."""
    from src.preprocessing import clean_dataframe
    df = load_raw_data()
    df = clean_dataframe(df)
    return df


@st.cache_data
def load_engineered_data() -> pd.DataFrame:
    """Load data with all feature engineering applied."""
    from src.feature_engineering import engineer_features
    df = load_clean_data()
    df = engineer_features(df)
    return df
