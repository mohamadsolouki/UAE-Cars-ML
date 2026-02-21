"""
Data preprocessing utilities for the UAE Used Cars dataset.
Handles cleaning, type conversion, and standardization.
"""

import pandas as pd
import numpy as np
import re


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw DataFrame.

    Steps:
    1. Strip whitespace from Location column
    2. Convert Cylinders to numeric (Unknown -> NaN)
    3. Title-case Make and Model for display
    4. Parse Description into Condition and Features
    5. Handle data quality issues
    """
    df = df.copy()

    # 1. Strip whitespace from Location
    df["Location"] = df["Location"].str.strip()

    # 2. Convert Cylinders to numeric
    df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")

    # Impute missing Cylinders with median by Make
    make_cyl_median = df.groupby("Make")["Cylinders"].transform("median")
    df["Cylinders"] = df["Cylinders"].fillna(make_cyl_median)
    # If still NaN (makes with all unknowns), fill with global median
    df["Cylinders"] = df["Cylinders"].fillna(df["Cylinders"].median())
    df["Cylinders"] = df["Cylinders"].astype(int)

    # 3. Title-case Make and Model for display
    df["Make_Display"] = df["Make"].str.replace("-", " ").str.title()
    df["Model_Display"] = df["Model"].str.replace("-", " ").str.title()

    # 4. Parse Description
    df["Condition"] = df["Description"].apply(_extract_condition)
    df["Features_List"] = df["Description"].apply(_extract_features)

    return df


def _extract_condition(description: str) -> str:
    """Extract the condition from the description text."""
    try:
        match = re.search(r"Condition:\s*(.+?)\.", description)
        if match:
            return match.group(1).strip()
    except (TypeError, AttributeError):
        pass
    return "Unknown"


def _extract_features(description: str) -> list:
    """Extract the list of features from the description text."""
    try:
        match = re.search(r"with (.+?)\.\s*Condition:", description)
        if match:
            return [f.strip() for f in match.group(1).split(",")]
    except (TypeError, AttributeError):
        pass
    return []


# All possible features found in the dataset
ALL_FEATURES = [
    "Sunroof",
    "Leather seats",
    "Navigation system",
    "Bluetooth",
    "Rear camera",
    "Adaptive cruise control",
]

# Condition severity ordering (lower = better condition)
CONDITION_ORDER = {
    "No damage": 0,
    "Minor scratches": 1,
    "Repainted bumper": 2,
    "Dented door": 3,
    "Engine repaired": 4,
    "Accident history": 5,
    "Unknown": 3,  # neutral default
}
