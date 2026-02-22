# UAE Used Cars -- Machine Learning Analysis

A comprehensive, end-to-end data science project analyzing 10,000 used car listings from the UAE market. Built with Streamlit, scikit-learn, XGBoost, and Plotly.

## Live Demo

[Launch the App on Streamlit Community Cloud](https://uae-cars-ml.streamlit.app)

## Overview

This application serves as both an analytical tool and educational material covering the full ML pipeline:

- **Data Overview** -- Interactive exploration of the raw dataset with quality assessment
- **Exploratory Analysis** -- 20+ interactive Plotly visualizations with market insights
- **Feature Engineering** -- Step-by-step walkthrough of all data transformations
- **Modeling** -- 4 ML tasks: price regression, body type classification, market segmentation, anomaly detection
- **Predictions** -- Interactive tools: price estimator, body type predictor, deal analyzer, segment finder

## Key Findings

| Insight | Detail |
|---|---|
| Market Structure | Two-tier: mass market (88%) vs luxury (12%) with 5.6x price difference |
| Top Price Driver | Number of cylinders (correlation: 0.45 with price) |
| Geographic Concentration | Dubai holds 80% of listings with 2.7x higher median price than Sharjah |
| Best Regression Model | Random Forest (R-squared: 0.55, CV: 0.58) |
| Best Classifier | Gradient Boosting (F1: 0.60 for body type) |
| Market Segments | 2 natural clusters found via K-Means (silhouette: 0.48) |

## ML Tasks

### Task 1: Price Prediction (Regression)
6 models compared: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost. Random Forest achieves the best generalization with CV R-squared of 0.58.

### Task 2: Body Type Classification
Predicts body type (SUV, Sedan, Coupe, etc.) from car attributes. Gradient Boosting achieves 61% accuracy across 6 classes.

### Task 3: Market Segmentation
K-Means clustering identifies 2 natural market segments: mass market (avg 159K AED) and luxury (avg 881K AED).

### Task 4: Anomaly Detection
Isolation Forest identifies 5% of listings as anomalous -- potential deals or overpriced vehicles.

## Setup

```bash
# Clone the repository
git clone https://github.com/fatehartin/UAE-Cars-ML.git
cd UAE-Cars-ML

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models from scratch
python train.py

# Launch the app
streamlit run home.py
```

## Project Structure

```
UAE-Cars-ML/
    .streamlit/config.toml        # Theme configuration
    src/
        data_loader.py            # Cached data loading
        preprocessing.py          # Cleaning and NLP extraction
        feature_engineering.py    # Feature transformations
        visualization.py          # Reusable Plotly charts
    pages/
        1_Data_Overview.py        # Dataset exploration
        2_Exploratory_Analysis.py # Visual EDA (20+ charts)
        3_Feature_Engineering.py  # Feature pipeline walkthrough
        4_Modeling.py             # Model results and analysis
        5_Predictions.py          # Interactive prediction tools
        6_About.py                # Documentation
    models/                       # Pre-trained model artifacts
    app.py                        # Main entry point
    train.py                      # Training pipeline
    requirements.txt              # Dependencies
    uae_used_cars_10k.csv         # Dataset (10K records)
```

## Tech Stack

- **Streamlit** -- Web application framework
- **pandas / NumPy** -- Data processing
- **Plotly** -- Interactive visualizations
- **scikit-learn** -- ML framework
- **XGBoost** -- Gradient boosted trees
- **joblib** -- Model serialization

## License

MIT License -- see [LICENSE](LICENSE) for details.