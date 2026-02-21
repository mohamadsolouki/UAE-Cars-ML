"""
Feature engineering utilities for the UAE Used Cars dataset.
Transforms cleaned data into model-ready features.
"""

import pandas as pd
import numpy as np
from src.preprocessing import ALL_FEATURES, CONDITION_ORDER


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Creates:
    - Car_Age: Age of the car (2025 - Year)
    - Mileage_Per_Year: Average annual mileage
    - Log_Price: Log-transformed price for modeling
    - Condition_Score: Ordinal encoding of condition severity
    - Binary feature columns from Description (has_sunroof, etc.)
    - Feature_Count: Total number of features per car
    - Is_Luxury: Whether the make is a luxury brand
    - Price_Per_Km: Price divided by mileage
    """
    df = df.copy()

    # Derived numerical features
    df["Car_Age"] = 2025 - df["Year"]
    df["Car_Age"] = df["Car_Age"].clip(lower=1)  # Avoid division by zero
    df["Mileage_Per_Year"] = df["Mileage"] / df["Car_Age"]
    df["Log_Price"] = np.log1p(df["Price"])
    df["Log_Mileage"] = np.log1p(df["Mileage"])
    df["Price_Per_Km"] = df["Price"] / df["Mileage"].replace(0, 1)

    # Condition severity score
    df["Condition_Score"] = df["Condition"].map(CONDITION_ORDER).fillna(3).astype(int)

    # Binary feature indicators from Description
    for feature in ALL_FEATURES:
        col_name = "has_" + feature.lower().replace(" ", "_")
        df[col_name] = df["Features_List"].apply(
            lambda x: 1 if feature in x else 0
        )

    # Feature count
    df["Feature_Count"] = df["Features_List"].apply(len)

    # Luxury brand indicator
    luxury_makes = [
        "ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
        "aston-martin", "mclaren", "maybach", "bugatti", "porsche",
    ]
    df["Is_Luxury"] = df["Make"].isin(luxury_makes).astype(int)

    return df


def get_model_features() -> list:
    """Return the list of feature column names used for modeling."""
    return [
        "Year", "Mileage", "Cylinders", "Car_Age", "Mileage_Per_Year",
        "Log_Mileage", "Condition_Score", "Feature_Count", "Is_Luxury",
        "has_sunroof", "has_leather_seats", "has_navigation_system",
        "has_bluetooth", "has_rear_camera", "has_adaptive_cruise_control",
    ]


def get_categorical_features() -> list:
    """Return categorical column names that need encoding."""
    return ["Make", "Body Type", "Transmission", "Fuel Type", "Color", "Location"]


def get_all_feature_names() -> list:
    """Return all feature names (numerical + categorical)."""
    return get_model_features() + get_categorical_features()
