"""
Training script for the UAE Used Cars ML project.
Trains all models, computes metrics, and saves artifacts to models/ directory.

Run: python train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, silhouette_score,
)
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_dataframe, CONDITION_ORDER, ALL_FEATURES
from src.feature_engineering import (
    engineer_features, get_model_features, get_categorical_features,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_and_prepare_data():
    """Load CSV, clean, and engineer features."""
    print("[1/7] Loading and preparing data...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "uae_used_cars_10k.csv"))
    df = clean_dataframe(df)
    df = engineer_features(df)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Price range: {df['Price'].min():,.0f} - {df['Price'].max():,.0f} AED")
    return df


def build_preprocessor():
    """Build a ColumnTransformer for all features."""
    numerical_features = get_model_features()
    categorical_features = get_categorical_features()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numerical_features, categorical_features


def mean_absolute_percentage_error(y_true, y_pred):
    """Compute MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_regression_models(df):
    """Train all price regression models."""
    print("\n[2/7] Training regression models for Price prediction...")

    preprocessor, num_feats, cat_feats = build_preprocessor()
    feature_cols = num_feats + cat_feats

    X = df[feature_cols].copy()
    y = df["Price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=100.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
    }

    results = {}
    best_r2 = -np.inf
    best_model_name = None

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_processed, y_train)
        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)

        # Compute metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test)

        # Cross-validation (5-fold)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model.__class__(**model.get_params())),
        ])
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)

        results[name] = {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        }

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"price_{safe_name}.joblib"))

        print(f"    Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | "
              f"RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | MAPE: {mape:.1f}%")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = name

    print(f"\n  Best model: {best_model_name} (Test R2: {best_r2:.4f})")

    # Save test data for visualization in app
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_processed)
    test_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_best,
    })
    test_results.to_csv(os.path.join(MODELS_DIR, "regression_test_results.csv"), index=False)

    # Feature importance from best tree model (either XGBoost, GB, or RF)
    tree_models = ["XGBoost", "Gradient Boosting", "Random Forest"]
    for tm_name in tree_models:
        if tm_name in models:
            tm = models[tm_name]
            importances = tm.feature_importances_
            all_feature_names = num_feats + cat_feats
            feat_imp = dict(zip(all_feature_names, importances.tolist()))
            joblib.dump(feat_imp, os.path.join(MODELS_DIR, f"feature_importance_{tm_name.lower().replace(' ', '_')}.joblib"))

    results["best_model"] = best_model_name
    return results, X_train, X_test, y_train, y_test, preprocessor


def train_classification_models(df):
    """Train body type classification models."""
    print("\n[3/7] Training classification models for Body Type prediction...")

    # Filter to top body types for meaningful classification
    top_body_types = df["Body Type"].value_counts().head(6).index.tolist()
    df_cls = df[df["Body Type"].isin(top_body_types)].copy()

    preprocessor, num_feats, cat_feats = build_preprocessor()
    # Remove Body Type from categorical features (it's the target)
    cls_cat_feats = [f for f in cat_feats if f != "Body Type"]
    feature_cols = num_feats + cls_cat_feats

    cls_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cls_cat_feats),
        ],
        remainder="drop",
    )

    X = df_cls[feature_cols].copy()
    y = df_cls["Body Type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_proc = cls_preprocessor.fit_transform(X_train)
    X_test_proc = cls_preprocessor.transform(X_test)

    # Save classifier preprocessor
    joblib.dump(cls_preprocessor, os.path.join(MODELS_DIR, "cls_preprocessor.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "cls_feature_columns.joblib"))
    joblib.dump(top_body_types, os.path.join(MODELS_DIR, "cls_labels.joblib"))

    models = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    best_f1 = -1
    best_cls_name = None

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted")
        prec_w = precision_score(y_test, y_pred, average="weighted")
        rec_w = recall_score(y_test, y_pred, average="weighted")

        cm = confusion_matrix(y_test, y_pred, labels=top_body_types)
        cls_report = classification_report(y_test, y_pred, output_dict=True)

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"bodytype_{safe_name}.joblib"))

        results[name] = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "precision_weighted": round(prec_w, 4),
            "recall_weighted": round(rec_w, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                k: v for k, v in cls_report.items()
                if isinstance(v, dict)
            },
        }

        print(f"    Accuracy: {acc:.4f} | F1 (weighted): {f1_w:.4f}")

        if f1_w > best_f1:
            best_f1 = f1_w
            best_cls_name = name

    results["best_model"] = best_cls_name
    print(f"\n  Best classifier: {best_cls_name} (F1: {best_f1:.4f})")
    return results


def train_clustering(df):
    """Perform K-Means clustering for market segmentation."""
    print("\n[4/7] Training clustering model for market segmentation...")

    cluster_features = ["Price", "Mileage", "Car_Age", "Cylinders", "Is_Luxury"]
    X_cluster = df[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Elbow method
    inertias = []
    silhouettes = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))
        print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={silhouettes[-1]:.4f}")

    # Choose optimal K (best silhouette)
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  Optimal K by silhouette: {best_k}")

    # Final model with optimal K
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Save artifacts
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "cluster_scaler.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "cluster_pca.joblib"))
    joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.joblib"))

    # Cluster profiles
    df_clustered = df[cluster_features].copy()
    df_clustered["Cluster"] = cluster_labels
    profiles = {}
    for c in range(best_k):
        mask = df_clustered["Cluster"] == c
        profile = {
            "count": int(mask.sum()),
            "avg_price": round(df_clustered.loc[mask, "Price"].mean(), 0),
            "avg_mileage": round(df_clustered.loc[mask, "Mileage"].mean(), 0),
            "avg_car_age": round(df_clustered.loc[mask, "Car_Age"].mean(), 1),
            "avg_cylinders": round(df_clustered.loc[mask, "Cylinders"].mean(), 1),
            "luxury_pct": round(df_clustered.loc[mask, "Is_Luxury"].mean() * 100, 1),
        }
        profiles[str(c)] = profile
        print(f"  Cluster {c}: n={profile['count']}, avg_price={profile['avg_price']:,.0f}, "
              f"avg_mileage={profile['avg_mileage']:,.0f}, luxury={profile['luxury_pct']:.1f}%")

    cluster_results = {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k": best_k,
        "profiles": profiles,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "pca_coords": X_pca.tolist(),
        "labels": cluster_labels.tolist(),
    }

    return cluster_results


def train_anomaly_detection(df):
    """Train Isolation Forest for price anomaly detection."""
    print("\n[5/7] Training anomaly detection model...")

    anomaly_features = ["Price", "Mileage", "Car_Age", "Cylinders", "Is_Luxury"]
    X_anom = df[anomaly_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anom)

    iso_forest = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    anomaly_labels = iso_forest.predict(X_scaled)  # -1 for anomaly, 1 for normal
    anomaly_scores = iso_forest.decision_function(X_scaled)

    # Save
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.joblib"))
    joblib.dump(anomaly_features, os.path.join(MODELS_DIR, "anomaly_features.joblib"))

    n_anomalies = (anomaly_labels == -1).sum()
    print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}% of data)")

    # Get top anomalies for display
    df_anom = df[["Make", "Model", "Year", "Price", "Mileage", "Body Type", "Location"]].copy()
    df_anom["Anomaly_Score"] = anomaly_scores
    df_anom["Is_Anomaly"] = anomaly_labels == -1
    df_anom.to_csv(os.path.join(MODELS_DIR, "anomaly_results.csv"), index=False)

    top_anomalies = df_anom[df_anom["Is_Anomaly"]].nsmallest(20, "Anomaly_Score")
    print("\n  Top anomalies (most unusual):")
    for _, row in top_anomalies.head(10).iterrows():
        print(f"    {row['Year']} {row['Make']} {row['Model']} - "
              f"{row['Price']:,.0f} AED, {row['Mileage']:,.0f} km, "
              f"Score: {row['Anomaly_Score']:.3f}")

    anomaly_results = {
        "n_anomalies": int(n_anomalies),
        "pct_anomalies": round(n_anomalies / len(df) * 100, 1),
        "contamination": 0.05,
    }

    return anomaly_results


def save_dataset_metadata(df):
    """Save dataset metadata for the app to use."""
    print("\n[6/7] Saving dataset metadata...")

    metadata = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "makes": sorted(df["Make"].unique().tolist()),
        "models_by_make": {},
        "body_types": sorted(df["Body Type"].unique().tolist()),
        "fuel_types": sorted(df["Fuel Type"].unique().tolist()),
        "colors": sorted(df["Color"].unique().tolist()),
        "locations": sorted(df["Location"].unique().tolist()),
        "transmissions": sorted(df["Transmission"].unique().tolist()),
        "year_range": [int(df["Year"].min()), int(df["Year"].max())],
        "cylinder_values": sorted(df["Cylinders"].unique().tolist()),
        "conditions": list(CONDITION_ORDER.keys()),
        "car_features": ALL_FEATURES,
    }

    # Models by make (for prediction dropdowns)
    for make in metadata["makes"]:
        models_list = sorted(df[df["Make"] == make]["Model"].unique().tolist())
        metadata["models_by_make"][make] = models_list

    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.joblib"))
    print(f"  Saved metadata: {len(metadata['makes'])} makes, "
          f"{sum(len(v) for v in metadata['models_by_make'].values())} models")
    return metadata


def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("UAE Used Cars ML - Training Pipeline")
    print("=" * 60)

    # Load data
    df = load_and_prepare_data()

    # Save metadata
    metadata = save_dataset_metadata(df)

    # Train regression models
    reg_results, X_train, X_test, y_train, y_test, preprocessor = train_regression_models(df)

    # Train classification models
    cls_results = train_classification_models(df)

    # Train clustering
    cluster_results = train_clustering(df)

    # Train anomaly detection
    anomaly_results = train_anomaly_detection(df)

    # Save all metrics as JSON
    print("\n[7/7] Saving all metrics...")
    all_metrics = {
        "regression": reg_results,
        "classification": cls_results,
        "clustering": cluster_results,
        "anomaly_detection": anomaly_results,
    }

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  All artifacts saved to {MODELS_DIR}/")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Summary
    print(f"\nRegression best model: {reg_results['best_model']}")
    print(f"  Test R2: {reg_results[reg_results['best_model']]['test_r2']}")
    print(f"  RMSE: {reg_results[reg_results['best_model']]['rmse']:,.0f} AED")

    print(f"\nClassification best model: {cls_results['best_model']}")
    print(f"  F1 (weighted): {cls_results[cls_results['best_model']]['f1_weighted']}")

    print(f"\nClustering: {cluster_results['optimal_k']} segments found")
    print(f"Anomalies: {anomaly_results['n_anomalies']} detected "
          f"({anomaly_results['pct_anomalies']}%)")

    return all_metrics


if __name__ == "__main__":
    main()
