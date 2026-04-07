"""
LightGBM Model - Load and Evaluate Trained Model
This script loads a pre-trained LightGBM model and evaluates it on test data.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def load_and_prepare_data():
    """Load CSV data and prepare features for modeling"""
    print("Loading data...")
    path = r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\Flood_Model\data\proceessed\flood_training_data_10k_clean.csv"
    df = pd.read_csv(path)

    print(f"Data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Feature division
    X = df.drop(["flood", "point_id", "easting", "northing"], axis=1)
    y = df["flood"]

    return X, y, df


def split_and_scale_data(X, y):
    """Split data and apply standardization"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Standardization
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def load_trained_model():
    """Load pre-trained LightGBM model from pickle file"""
    print("\nLoading trained model...")
    model_path = r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\Flood_Model\model_training\lgbm\best_lgbm_model.pkl"

    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {model_path}")
        return None


def make_predictions(model, X_test):
    """Generate predictions on test data"""
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"✓ Predictions generated (shape: {y_pred.shape})")
    return y_pred, y_pred_proba


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    print("\n" + "=" * 50)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 50)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"ROC AUC:       {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    """Plot ROC curve"""
    print("\nGenerating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"LightGBM (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("LightGBM ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✓ ROC curve displayed")


def plot_feature_importance(model, X):
    """Plot top 10 feature importance"""
    print("\nGenerating feature importance plot...")

    feature_importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    print("\nTop 10 Features:")
    print(feature_importance.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance["importance"])
    plt.yticks(range(len(feature_importance)), feature_importance["feature"])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Top 10 Feature Importance - LightGBM", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    print("✓ Feature importance plot displayed")

    return feature_importance


def plot_spatial_distribution(df):
    """Plot flood vs non-flood spatial distribution"""
    print("\nGenerating spatial distribution plot...")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["easting"], df["northing"], c=df["flood"], cmap="coolwarm", s=5)
    plt.xlabel("Easting", fontsize=12)
    plt.ylabel("Northing", fontsize=12)
    plt.title("Flood vs Non-Flood Spatial Distribution", fontsize=14, fontweight="bold")
    plt.colorbar(label="Flood (0=No, 1=Yes)")
    plt.tight_layout()
    plt.show()
    print("✓ Spatial distribution plot displayed")


def main():
    """Main execution pipeline"""
    print("=" * 50)
    print("LIGHTGBM MODEL - EVALUATION PIPELINE")
    print("=" * 50)

    # Load and prepare data
    X, y, df = load_and_prepare_data()

    # Plot spatial distribution
    plot_spatial_distribution(df)

    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Load trained model
    model = load_trained_model()
    if model is None:
        print("✗ Failed to load model. Exiting.")
        return

    # Make predictions
    y_pred, y_pred_proba = make_predictions(model, X_test)

    # Calculate and display metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, metrics["roc_auc"])

    # Plot feature importance
    feature_importance = plot_feature_importance(model, X)

    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
