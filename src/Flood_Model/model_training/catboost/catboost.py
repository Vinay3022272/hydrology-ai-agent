"""Notebook-to-script version of CatBoost evaluation workflow."""

import importlib
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _safe_joblib_load(model_path):
    """Load pickled model while avoiding local module shadowing issues."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    removed_paths = []

    for p in ["", script_dir, cwd]:
        while p in sys.path:
            sys.path.remove(p)
            removed_paths.append(p)

    try:
        importlib.invalidate_caches()
        importlib.import_module("catboost")
        return joblib.load(model_path)
    finally:
        sys.path.extend(removed_paths)


def main():
    print("=" * 50)
    print("CATBOOST MODEL - EVALUATION PIPELINE")
    print("=" * 50)

    # Cell 1: Load data
    print("Loading data...")
    path = r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\Flood_Model\data\proceessed\flood_training_data_10k_clean.csv"
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Cell 2: Spatial plot
    print("\nGenerating spatial distribution plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["easting"],
        df["northing"],
        c=df["flood"],
        cmap="coolwarm",
        s=5,
    )
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.title("Flood vs Non-Flood Spatial Distribution")
    plt.colorbar(label="Flood (0=No, 1=Yes)")
    plt.tight_layout()
    plt.show()
    print(" Spatial distribution plot displayed")

    # Cell 3: Feature division
    X = df.drop(["flood", "point_id", "easting", "northing"], axis=1)
    y = df["flood"]

    # Cell 4: Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Cell 5: Standardization
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cell 6: Shape print
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Cells 7-12: Model/grid setup from notebook (kept for parity, not executed)
    from catboost import CatBoostClassifier

    cat_model = CatBoostClassifier(verbose=0)
    param_grid_cat = {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7],
        "border_count": [32, 64, 128],
        "scale_pos_weight": [1, 5, 10],
    }

    grid_cat = GridSearchCV(
        estimator=cat_model,
        param_grid=param_grid_cat,
        cv=5,
        scoring="f1",
        verbose=2,
        n_jobs=-1,
    )

    # Cell 13: Load trained model
    print("\nLoading trained model...")
    cat_model_path = r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\Flood_Model\model_training\catboost\best_cat_model.pkl"
    model = _safe_joblib_load(cat_model_path)
    best_cat = model
    print(f" Model loaded successfully from {cat_model_path}")

    # Cell 14: Predictions
    print("\nGenerating predictions...")
    y_pred_cat = best_cat.predict(X_test)
    y_pred_proba_cat = best_cat.predict_proba(X_test)[:, 1]
    print(f" Predictions generated (shape: {y_pred_cat.shape})")

    # Cells 15-16: Metrics
    precision = precision_score(y_test, y_pred_cat)
    recall = recall_score(y_test, y_pred_cat)
    acc = accuracy_score(y_test, y_pred_cat)
    f1 = f1_score(y_test, y_pred_cat)
    roc_auc = roc_auc_score(y_test, y_pred_proba_cat)
    cm = confusion_matrix(y_test, y_pred_cat)

    print("\n" + "=" * 50)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"ROC AUC:       {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred_cat))

    # Cell 17: ROC
    print("\nGenerating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_cat)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"CatBoost (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Catboost ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(" ROC curve displayed")

    # Cell 18: Feature importance
    print("\nGenerating feature importance plot...")
    importances = np.asarray(model.feature_importances_)
    feature_names = list(X.columns)

    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        print(
            " Warning: feature name count and importance length differ "
            f"({len(feature_names)} vs {len(importances)}). "
            f"Using first {min_len} entries for plotting."
        )
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
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
    plt.title("Top 10 Feature Importance - Catboost", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    print(" Feature importance plot displayed")

    # Cell 19: SHAP summary (optional)
    try:
        import shap

        print("\nGenerating SHAP summary plot...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
        print(" SHAP summary plot displayed")
    except Exception as exc:
        print(f" SHAP step skipped: {exc}")

    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
