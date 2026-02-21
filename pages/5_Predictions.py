"""
Page 5: Predictions
Interactive tools for price prediction, body type classification,
market segment assignment, and deal quality assessment.
All using pre-trained models loaded from the models/ directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import CONDITION_ORDER, ALL_FEATURES
from src.feature_engineering import get_model_features, get_categorical_features
from src.visualization import apply_layout, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_WARN

st.set_page_config(
    page_title="Predictions | UAE Cars ML", layout="wide", page_icon="--"
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


@st.cache_data
def load_json(name):
    with open(os.path.join(MODELS_DIR, name), "r") as f:
        return json.load(f)


# Load all artifacts
metadata = load_model("metadata.joblib")
reg_preprocessor = load_model("preprocessor.joblib")
reg_feature_cols = load_model("feature_columns.joblib")
cls_preprocessor = load_model("cls_preprocessor.joblib")
cls_feature_cols = load_model("cls_feature_columns.joblib")
cls_labels = load_model("cls_labels.joblib")
cluster_scaler = load_model("cluster_scaler.joblib")
cluster_features_list = load_model("cluster_features.joblib")
kmeans = load_model("kmeans_model.joblib")
cluster_pca = load_model("cluster_pca.joblib")
anomaly_scaler = load_model("anomaly_scaler.joblib")
anomaly_features_list = load_model("anomaly_features.joblib")
iso_forest = load_model("isolation_forest.joblib")
metrics_data = load_json("metrics.json")

# Load regression models
best_reg_name = metrics_data["regression"]["best_model"]
best_reg_key = "price_" + best_reg_name.lower().replace(" ", "_") + ".joblib"
reg_model = load_model(best_reg_key)

# Check if model uses log-transformed target
uses_log_transform = metrics_data["regression"].get(best_reg_name, {}).get("uses_log_transform", False)

# Load classification models
best_cls_name = metrics_data["classification"]["best_model"]
best_cls_key = "bodytype_" + best_cls_name.lower().replace(" ", "_") + ".joblib"
cls_model = load_model(best_cls_key)


def validate_car_configuration(make, body_type, cylinders):
    """
    Validate if a car configuration is plausible based on historical data.
    Returns (is_valid, warning_messages).
    """
    warnings = []
    is_plausible = True
    
    # Get validation constraints from metadata
    valid_cyls_by_bt = metadata.get("valid_cylinders_by_body_type", {})
    valid_cyls_by_make = metadata.get("valid_cylinders_by_make", {})
    luxury_makes = metadata.get("luxury_makes", [])
    
    # Check if cylinders are valid for body type
    if body_type in valid_cyls_by_bt:
        valid_cyls_bt = valid_cyls_by_bt[body_type]
        if cylinders not in valid_cyls_bt:
            warnings.append(
                f"**Unusual Configuration:** {cylinders}-cylinder engines are rare for {body_type}s. "
                f"Typical options: {', '.join(str(c) for c in sorted(valid_cyls_bt))} cylinders."
            )
            is_plausible = False
    
    # Check if cylinders are valid for make
    if make in valid_cyls_by_make:
        valid_cyls_make = valid_cyls_by_make[make]
        if cylinders not in valid_cyls_make:
            make_display = make.replace('-', ' ').title()
            warnings.append(
                f"**Unusual for Brand:** {make_display} typically does not offer {cylinders}-cylinder engines. "
                f"Common options: {', '.join(str(c) for c in sorted(valid_cyls_make))} cylinders."
            )
            is_plausible = False
    
    # Special check for luxury cars with low cylinders
    if make in luxury_makes and cylinders < 6:
        warnings.append(
            f"**Luxury Brand Alert:** {make.replace('-', ' ').title()} vehicles typically have 6+ cylinder engines. "
            f"A {cylinders}-cylinder configuration may not exist for this brand."
        )
        is_plausible = False
    
    # Check for economy cars with high cylinders
    economy_makes = ["hyundai", "kia", "nissan", "toyota", "honda", "mazda", "mitsubishi", "suzuki"]
    if make in economy_makes and cylinders >= 10:
        warnings.append(
            f"**Configuration Warning:** {make.replace('-', ' ').title()} typically does not offer {cylinders}-cylinder engines. "
            f"This may produce unrealistic estimates."
        )
        is_plausible = False
    
    return is_plausible, warnings


def build_input_row(
    make, model, year, mileage, body_type, cylinders,
    transmission, fuel_type, color, location, condition, features_selected
):
    """Build a feature DataFrame from user inputs."""
    car_age = max(2025 - year, 1)
    mileage_per_year = mileage / car_age
    log_mileage = np.log1p(mileage)
    condition_score = CONDITION_ORDER.get(condition, 3)
    feature_count = len(features_selected)
    is_luxury = 1 if make in [
        "ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
        "aston-martin", "mclaren", "maybach", "porsche", "bugatti",
    ] else 0

    row = {
        "Year": year,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Car_Age": car_age,
        "Mileage_Per_Year": mileage_per_year,
        "Log_Mileage": log_mileage,
        "Condition_Score": condition_score,
        "Feature_Count": feature_count,
        "Is_Luxury": is_luxury,
    }

    for feat in ALL_FEATURES:
        col_name = "has_" + feat.lower().replace(" ", "_")
        row[col_name] = 1 if feat in features_selected else 0

    row["Make"] = make
    row["Body Type"] = body_type
    row["Transmission"] = transmission
    row["Fuel Type"] = fuel_type
    row["Color"] = color
    row["Location"] = location

    return row


def render_input_form(key_prefix="pred"):
    """Render the car input form and return values."""
    col1, col2, col3 = st.columns(3)

    with col1:
        make = st.selectbox(
            "Make (Brand)",
            options=metadata["makes"],
            format_func=lambda x: x.replace("-", " ").title(),
            key=f"{key_prefix}_make",
        )
        models_list = metadata["models_by_make"].get(make, ["other"])
        model = st.selectbox(
            "Model",
            options=models_list,
            format_func=lambda x: x.replace("-", " ").title(),
            key=f"{key_prefix}_model",
        )
        year = st.slider(
            "Year",
            min_value=metadata["year_range"][0],
            max_value=metadata["year_range"][1],
            value=2018,
            key=f"{key_prefix}_year",
        )
        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            max_value=500000,
            value=80000,
            step=5000,
            key=f"{key_prefix}_mileage",
        )

    with col2:
        body_type = st.selectbox(
            "Body Type",
            options=metadata["body_types"],
            key=f"{key_prefix}_bt",
        )
        cylinders = st.selectbox(
            "Cylinders",
            options=sorted([int(c) for c in metadata["cylinder_values"]]),
            index=1,
            key=f"{key_prefix}_cyl",
        )
        transmission = st.selectbox(
            "Transmission",
            options=metadata["transmissions"],
            key=f"{key_prefix}_trans",
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            options=metadata["fuel_types"],
            key=f"{key_prefix}_fuel",
        )

    with col3:
        color = st.selectbox(
            "Color",
            options=metadata["colors"],
            key=f"{key_prefix}_color",
        )
        location = st.selectbox(
            "Location (Emirate)",
            options=metadata["locations"],
            key=f"{key_prefix}_loc",
        )
        condition = st.selectbox(
            "Condition",
            options=[c for c in metadata["conditions"] if c != "Unknown"],
            key=f"{key_prefix}_cond",
        )
        features_selected = st.multiselect(
            "Car Features",
            options=ALL_FEATURES,
            default=[],
            key=f"{key_prefix}_feats",
        )

    return make, model, year, mileage, body_type, cylinders, transmission, fuel_type, color, location, condition, features_selected


st.title("Interactive Predictions")
st.markdown(
    "Use the pre-trained models to make predictions on custom car configurations. "
    "Select the tool below and configure the car attributes to see results."
)
st.markdown("---")

# Tabs for different prediction tools
tab1, tab2, tab3, tab4 = st.tabs([
    "Price Predictor",
    "Body Type Predictor",
    "Deal Analyzer",
    "Market Segment Finder",
])

# ============================================================
# Tab 1: Price Prediction
# ============================================================
with tab1:
    st.header("Price Predictor")
    
    # Get R2 display (use original scale if available)
    reg_metrics = metrics_data['regression'][best_reg_name]
    display_r2 = reg_metrics.get('test_r2_original_scale', reg_metrics.get('test_r2', 0))
    log_note = " (log-transformed target)" if reg_metrics.get('uses_log_transform', False) else ""
    
    st.markdown(f"""
    Estimate the market value of a used car using the **{best_reg_name}** model{log_note}
    (Test R-squared: {display_r2:.4f}, 
    CV: {metrics_data['regression'][best_reg_name]['cv_mean']:.4f}).
    
    Configure the car attributes below and click "Predict Price" to see the estimated value.
    
    **Note:** The model validates your configuration against historical data to warn about 
    unusual combinations (e.g., sedan with 12 cylinders) that may produce unrealistic estimates.
    """)

    make, model, year, mileage, body_type, cylinders, transmission, fuel_type, color, location, condition, features_selected = render_input_form("price")

    if st.button("Predict Price", type="primary", key="btn_price"):
        # Validate configuration
        is_plausible, validation_warnings = validate_car_configuration(make, body_type, cylinders)
        
        row = build_input_row(
            make, model, year, mileage, body_type, cylinders,
            transmission, fuel_type, color, location, condition, features_selected
        )
        input_df = pd.DataFrame([row])
        input_processed = reg_preprocessor.transform(input_df[reg_feature_cols])
        
        # Get prediction - handle log-transformed models
        predicted_raw = reg_model.predict(input_processed)[0]
        if uses_log_transform:
            # Convert from log-space back to original price
            predicted_price = np.expm1(predicted_raw)
        else:
            predicted_price = predicted_raw
        predicted_price = max(predicted_price, 0)

        st.markdown("---")
        
        # Show validation warnings first
        if validation_warnings:
            st.warning("**Configuration Validation Issues Detected:**")
            for warning in validation_warnings:
                st.markdown(f"- {warning}")
            st.markdown("---")
        
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"{predicted_price:,.0f} AED")
        col2.metric("Car Age", f"{max(2025 - year, 1)} years")
        col3.metric(
            "Segment",
            "Luxury" if row["Is_Luxury"] else "Mass Market"
        )
        
        # Confidence indicator based on configuration validity
        if not is_plausible:
            st.error(
                "**Low Confidence Estimate:** The selected configuration is unusual and may not exist "
                "in the real market. The predicted price should be treated with caution."
            )

        st.markdown(f"""
        **Estimation Details:**
        - Model used: {best_reg_name}{" (using log-transformed prices for better accuracy)" if uses_log_transform else ""}
        - The predicted price is based on {len(reg_feature_cols)} features
        - Key factors for this prediction: **{make.replace('-', ' ').title()}** brand, 
          **{cylinders}** cylinders, **{body_type}**, **{year}** model year
        
        **Important disclaimer:** This is a model estimate based on historical listing data. 
        Actual market value depends on many factors not captured in the data, including 
        specific trim level, service history, dealer vs private sale, and current market conditions.
        """)

        # Context: show similar cars in dataset
        from src.data_loader import load_clean_data
        df_ref = load_clean_data()
        similar = df_ref[
            (df_ref["Make"] == make) &
            (df_ref["Year"].between(year - 2, year + 2))
        ]["Price"]

        if len(similar) > 0:
            st.markdown(f"""
            **Market Context:** Among {len(similar)} similar {make.replace('-', ' ').title()} 
            cars (years {year-2}-{year+2}) in the dataset:
            - Minimum: {similar.min():,.0f} AED
            - Median: {similar.median():,.0f} AED
            - Maximum: {similar.max():,.0f} AED
            - Your estimate: {predicted_price:,.0f} AED
            """)

# ============================================================
# Tab 2: Body Type Classification
# ============================================================
with tab2:
    st.header("Body Type Predictor")
    st.markdown(f"""
    Predict the most likely body type for a car configuration using the 
    **{best_cls_name}** model (Accuracy: {metrics_data['classification'][best_cls_name]['accuracy']:.4f}).
    
    This tool is useful for automated categorization of new listings or verifying 
    existing classifications.
    """)

    make2, model2, year2, mileage2, body_type2, cylinders2, transmission2, fuel_type2, color2, location2, condition2, features_selected2 = render_input_form("cls")

    if st.button("Predict Body Type", type="primary", key="btn_cls"):
        row = build_input_row(
            make2, model2, year2, mileage2, body_type2, cylinders2,
            transmission2, fuel_type2, color2, location2, condition2, features_selected2
        )
        input_df = pd.DataFrame([row])

        # Use only classification features (exclude Body Type)
        input_processed = cls_preprocessor.transform(input_df[cls_feature_cols])
        predicted_class = cls_model.predict(input_processed)[0]

        # Get probabilities if available
        if hasattr(cls_model, "predict_proba"):
            probas = cls_model.predict_proba(input_processed)[0]
            proba_df = pd.DataFrame({
                "Body Type": cls_model.classes_,
                "Probability": probas,
            }).sort_values("Probability", ascending=True)

        st.markdown("---")
        st.subheader("Classification Result")

        st.markdown(f"**Predicted Body Type: {predicted_class}**")

        if hasattr(cls_model, "predict_proba"):
            fig = px.bar(
                proba_df, x="Probability", y="Body Type",
                orientation="h",
                color="Probability",
                color_continuous_scale="Blues",
            )
            fig = apply_layout(fig, "Prediction Probabilities by Body Type", height=350)
            st.plotly_chart(fig, use_container_width=True)

            top_prob = proba_df.iloc[-1]
            st.markdown(f"""
            The model predicts **{predicted_class}** with {top_prob['Probability']*100:.1f}% confidence.
            
            The probability distribution shows how certain the model is about its prediction. 
            A dominant bar indicates high confidence; evenly distributed bars suggest the 
            model finds multiple body types plausible for this configuration.
            """)

# ============================================================
# Tab 3: Deal Analyzer
# ============================================================
with tab3:
    st.header("Deal Analyzer")
    st.markdown("""
    Assess whether a car listing is fairly priced, overpriced, or a good deal using 
    anomaly detection. Enter a car's details including its listed price, and the model 
    will evaluate it against market norms.
    """)

    make3, model3, year3, mileage3, body_type3, cylinders3, transmission3, fuel_type3, color3, location3, condition3, features_selected3 = render_input_form("deal")

    listed_price = st.number_input(
        "Listed Price (AED)",
        min_value=1000,
        max_value=20000000,
        value=150000,
        step=5000,
        key="deal_price",
    )

    if st.button("Analyze Deal", type="primary", key="btn_deal"):
        row = build_input_row(
            make3, model3, year3, mileage3, body_type3, cylinders3,
            transmission3, fuel_type3, color3, location3, condition3, features_selected3
        )

        # Get predicted price for comparison (handle log-transformed models)
        input_df = pd.DataFrame([row])
        input_processed = reg_preprocessor.transform(input_df[reg_feature_cols])
        predicted_raw = reg_model.predict(input_processed)[0]
        if uses_log_transform:
            predicted_price = np.expm1(predicted_raw)
        else:
            predicted_price = predicted_raw
        predicted_price = max(predicted_price, 0)

        # Get anomaly score
        car_age = max(2025 - year3, 1)
        is_luxury = 1 if make3 in [
            "ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
            "aston-martin", "mclaren", "maybach", "porsche", "bugatti",
        ] else 0

        anomaly_input = pd.DataFrame([{
            "Price": listed_price,
            "Mileage": mileage3,
            "Car_Age": car_age,
            "Cylinders": cylinders3,
            "Is_Luxury": is_luxury,
        }])
        anomaly_scaled = anomaly_scaler.transform(anomaly_input[anomaly_features_list])
        anomaly_score = iso_forest.decision_function(anomaly_scaled)[0]
        is_anomaly = iso_forest.predict(anomaly_scaled)[0] == -1

        st.markdown("---")
        st.subheader("Deal Assessment")

        price_diff = listed_price - predicted_price
        price_diff_pct = (price_diff / predicted_price * 100) if predicted_price > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Listed Price", f"{listed_price:,.0f} AED")
        col2.metric("Estimated Value", f"{predicted_price:,.0f} AED")
        col3.metric("Difference", f"{price_diff:+,.0f} AED", delta=f"{price_diff_pct:+.1f}%")
        col4.metric("Anomaly Score", f"{anomaly_score:.3f}")

        # Deal rating
        if price_diff_pct < -15:
            deal_rating = "Excellent Deal"
            deal_color = "#2ca02c"
            deal_msg = "This listing is significantly below the estimated market value. If the car is in the described condition, this represents a strong buying opportunity."
        elif price_diff_pct < -5:
            deal_rating = "Good Deal"
            deal_color = "#7ec850"
            deal_msg = "This listing is moderately below market value. Worth considering, though the discount may reflect factors not captured in the model."
        elif price_diff_pct < 5:
            deal_rating = "Fair Price"
            deal_color = "#ffa500"
            deal_msg = "This listing is priced close to the estimated market value. The price appears reasonable given the car's specifications."
        elif price_diff_pct < 15:
            deal_rating = "Slightly Overpriced"
            deal_color = "#ff6347"
            deal_msg = "This listing is moderately above market value. There may be room for negotiation."
        else:
            deal_rating = "Overpriced"
            deal_color = "#d62728"
            deal_msg = "This listing is significantly above the estimated market value. Consider negotiating or looking at alternative listings."

        st.markdown(
            f'<div style="background-color:{deal_color}22; border-left: 4px solid {deal_color}; '
            f'padding: 1rem; border-radius: 5px; margin: 1rem 0;">'
            f'<strong style="color:{deal_color}; font-size: 1.3rem;">{deal_rating}</strong><br>'
            f'{deal_msg}</div>',
            unsafe_allow_html=True,
        )

        if is_anomaly:
            st.warning(
                "This listing was flagged as anomalous by the Isolation Forest model "
                "(anomaly score < 0). The combination of price, mileage, age, and cylinders "
                "is unusual compared to similar cars in the market."
            )

        # Similar cars in market
        from src.data_loader import load_clean_data
        df_ref = load_clean_data()
        similar = df_ref[
            (df_ref["Make"] == make3) &
            (df_ref["Body Type"] == body_type3) &
            (df_ref["Year"].between(year3 - 3, year3 + 3))
        ]

        if len(similar) > 2:
            st.markdown(f"**Comparable listings in dataset** ({len(similar)} found):")
            fig = go.Figure()
            fig.add_trace(go.Box(
                x=similar["Price"], name="Market Range",
                marker_color=COLOR_PRIMARY, boxmean=True,
            ))
            fig.add_vline(
                x=listed_price, line_dash="dash", line_color=COLOR_WARN,
                annotation_text=f"Listed: {listed_price:,.0f}",
            )
            fig.add_vline(
                x=predicted_price, line_dash="dot", line_color=COLOR_ACCENT,
                annotation_text=f"Estimated: {predicted_price:,.0f}",
            )
            fig.update_layout(xaxis_title="Price (AED)")
            fig = apply_layout(fig, "Your Listing vs Market Range", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 4: Market Segment Finder
# ============================================================
with tab4:
    st.header("Market Segment Finder")

    cluster_metrics = metrics_data["clustering"]
    profiles = cluster_metrics["profiles"]

    st.markdown(f"""
    Classify a car into one of the {cluster_metrics['optimal_k']} identified market segments. 
    This helps understand where a particular car fits in the broader UAE used car market landscape.
    """)

    make4, model4, year4, mileage4, body_type4, cylinders4, transmission4, fuel_type4, color4, location4, condition4, features_selected4 = render_input_form("seg")

    if st.button("Find Segment", type="primary", key="btn_seg"):
        car_age = max(2025 - year4, 1)
        is_luxury = 1 if make4 in [
            "ferrari", "lamborghini", "rolls-royce", "bentley", "maserati",
            "aston-martin", "mclaren", "maybach", "porsche", "bugatti",
        ] else 0

        # Get predicted price for the car (handle log-transformed models)
        row = build_input_row(
            make4, model4, year4, mileage4, body_type4, cylinders4,
            transmission4, fuel_type4, color4, location4, condition4, features_selected4
        )
        input_df = pd.DataFrame([row])
        input_processed = reg_preprocessor.transform(input_df[reg_feature_cols])
        predicted_raw_seg = reg_model.predict(input_processed)[0]
        if uses_log_transform:
            predicted_price_seg = np.expm1(predicted_raw_seg)
        else:
            predicted_price_seg = predicted_raw_seg
        predicted_price_seg = max(predicted_price_seg, 0)

        cluster_input = pd.DataFrame([{
            "Price": predicted_price_seg,
            "Mileage": mileage4,
            "Car_Age": car_age,
            "Cylinders": cylinders4,
            "Is_Luxury": is_luxury,
        }])
        cluster_scaled = cluster_scaler.transform(cluster_input[cluster_features_list])
        cluster_label = kmeans.predict(cluster_scaled)[0]
        pca_point = cluster_pca.transform(cluster_scaled)[0]

        st.markdown("---")
        st.subheader("Segment Assignment")

        profile = profiles[str(cluster_label)]

        # Determine segment names
        segment_names = {}
        for cid, p in profiles.items():
            if p["luxury_pct"] > 50:
                segment_names[cid] = "Luxury Segment"
            else:
                segment_names[cid] = "Mass Market Segment"

        segment_name = segment_names.get(str(cluster_label), f"Segment {cluster_label}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Assigned Segment", segment_name)
        col2.metric("Segment Size", f"{profile['count']:,} cars")
        col3.metric("Est. Price", f"{predicted_price_seg:,.0f} AED")

        st.markdown(f"""
        **Segment Profile:**
        - Average price in segment: {profile['avg_price']:,.0f} AED
        - Average mileage: {profile['avg_mileage']:,.0f} km
        - Average car age: {profile['avg_car_age']:.1f} years
        - Average cylinders: {profile['avg_cylinders']:.1f}
        - Luxury brand composition: {profile['luxury_pct']:.1f}%
        """)

        # Show on PCA plot
        pca_coords = np.array(cluster_metrics["pca_coords"])
        labels = np.array(cluster_metrics["labels"])

        fig = go.Figure()
        for c in range(cluster_metrics["optimal_k"]):
            mask = labels == c
            fig.add_trace(go.Scatter(
                x=pca_coords[mask, 0], y=pca_coords[mask, 1],
                mode="markers",
                marker=dict(size=3, opacity=0.3),
                name=segment_names.get(str(c), f"Cluster {c}"),
            ))
        fig.add_trace(go.Scatter(
            x=[pca_point[0]], y=[pca_point[1]],
            mode="markers",
            marker=dict(size=15, color=COLOR_WARN, symbol="star",
                        line=dict(width=2, color="black")),
            name="Your Car",
        ))
        fig.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
        )
        fig = apply_layout(fig, "Your Car in Market Segment Space (PCA)", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        The star marker shows where your car falls in the PCA-reduced market space. 
        Cars closer together in this visualization have more similar overall profiles 
        (price, mileage, age, cylinders, and luxury status combined).
        """)
