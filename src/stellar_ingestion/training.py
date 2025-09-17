"""
Stellar Classification Model Training Pipeline

This module provides functions for training, evaluating, and saving
machine learning models for stellar classification.

Author: MLOps Team
Date: September 2025
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = sns = None

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


def load_config(config_path="config/datasets/stellar.yaml"):
    """Load stellar classification configuration"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def load_training_data(artifacts_dir="data/processed"):
    """Load preprocessed training data"""
    logger.info("Loading training data...")

    X_train = pd.read_csv(f"{artifacts_dir}/X_train.csv")
    X_test = pd.read_csv(f"{artifacts_dir}/X_test.csv")
    y_train = pd.read_csv(f"{artifacts_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{artifacts_dir}/y_test.csv").squeeze()
    scaler = joblib.load(f"{artifacts_dir}/scaler.pkl")

    logger.info(f"Training data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


def train_svm_model(X_train, y_train, config):
    """Train SVM model with parameters from config"""
    logger.info("Training SVM model...")

    # Use subset for SVM due to computational complexity
    from sklearn.model_selection import train_test_split

    if len(X_train) > 10000:
        logger.info(
            f"Using subset of 10000 samples for SVM training (original: {len(X_train)})"
        )
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train, train_size=10000, stratify=y_train, random_state=42
        )
    else:
        X_train_subset, y_train_subset = X_train, y_train

    svm_config = config["models"]["svm"]
    model = SVC(
        kernel=svm_config["kernel"],
        C=svm_config["C"],
        random_state=svm_config["random_state"],
        probability=True,
    )

    model.fit(X_train_subset, y_train_subset)
    logger.info("SVM model training completed")
    return model


def train_random_forest_model(X_train, y_train, config):
    """Train Random Forest model with parameters from config"""
    logger.info("Training Random Forest model...")

    rf_config = config["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_config["n_estimators"], random_state=rf_config["random_state"]
    )

    model.fit(X_train, y_train)
    logger.info("Random Forest model training completed")
    return model


def evaluate_model(model, X_test, y_test, model_name, config, save_plots=True):
    """Evaluate model performance"""
    logger.info(f"Evaluating {model_name} model...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    )

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")

    # Classification report
    class_names = config["classes"]["labels"]
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    logger.info(f"{model_name} Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    if save_plots:
        save_evaluation_plots(cm, model_name, class_names, y_test, y_pred_proba)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
    }


def save_evaluation_plots(cm, model_name, class_names, y_test, y_pred_proba=None):
    """Save evaluation plots"""
    plots_dir = "data/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ROC Curve for multiclass (if probabilities available)
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize

            # Binarize the output
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            n_classes = y_test_bin.shape[1]

            plt.figure(figsize=(10, 8))
            colors = ["blue", "red", "green"]

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    color=colors[i],
                    lw=2,
                    label=f"{class_names[i]} (AUC = {roc_auc:.2f})",
                )

            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves - {model_name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(
                f"{plots_dir}/roc_curves_{model_name.lower().replace(' ', '_')}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Could not create ROC curve: {e}")


def hyperparameter_tuning(X_train, y_train, model_type="random_forest"):
    """Perform hyperparameter tuning"""
    logger.info(f"Performing hyperparameter tuning for {model_type}...")

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
    elif model_type == "svm":
        model = SVC(random_state=42, probability=True)
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def save_model_artifacts(
    model, model_name, scaler, feature_names, config, models_dir="models"
):
    """Save model and related artifacts"""
    logger.info(f"Saving {model_name} model artifacts...")

    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = f"{models_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = f"{models_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Save feature names
    feature_names_path = f"{models_dir}/feature_names.json"
    import json

    with open(feature_names_path, "w") as f:
        json.dump(list(feature_names), f)

    # Save model info
    model_info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "classes": config["classes"]["labels"],
    }

    model_info_path = f"{models_dir}/model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Model artifacts saved to {models_dir}")
    return model_path


def train_stellar_models(
    config_path="config/datasets/stellar.yaml", artifacts_dir="data/temp"
):
    """Train all stellar classification models"""
    logger.info("Starting stellar classification model training...")

    # Load configuration
    config = load_config(config_path)

    # Set MLflow experiment
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    # Load training data
    X_train, X_test, y_train, y_test, scaler = load_training_data(artifacts_dir)

    models = {}
    results = {}

    with mlflow.start_run(run_name="stellar_classification_training"):

        # Log dataset information
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("n_features", X_train.shape[1])
        mlflow.log_metric("n_classes", len(config["classes"]["labels"]))

        # Train SVM Model
        logger.info("=" * 50)
        logger.info("Training SVM Model")
        logger.info("=" * 50)

        svm_model = train_svm_model(X_train, y_train, config)
        svm_results = evaluate_model(svm_model, X_test, y_test, "SVM", config)

        models["svm"] = svm_model
        results["svm"] = svm_results

        # Log SVM metrics
        mlflow.log_metric("svm_accuracy", svm_results["accuracy"])
        mlflow.log_metric(
            "svm_f1_macro",
            svm_results["classification_report"]["macro avg"]["f1-score"],
        )

        # Train Random Forest Model
        logger.info("=" * 50)
        logger.info("Training Random Forest Model")
        logger.info("=" * 50)

        rf_model = train_random_forest_model(X_train, y_train, config)
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", config)

        models["random_forest"] = rf_model
        results["random_forest"] = rf_results

        # Log Random Forest metrics
        mlflow.log_metric("rf_accuracy", rf_results["accuracy"])
        mlflow.log_metric(
            "rf_f1_macro", rf_results["classification_report"]["macro avg"]["f1-score"]
        )

        # Determine best model
        best_model_name = (
            "svm"
            if svm_results["accuracy"] > rf_results["accuracy"]
            else "random_forest"
        )
        best_model = models[best_model_name]
        best_accuracy = results[best_model_name]["accuracy"]

        logger.info(
            f"Best model: {best_model_name.upper()} with accuracy: {best_accuracy:.4f}"
        )
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_model", best_model_name)

        # Save best model artifacts
        model_path = save_model_artifacts(
            best_model, f"best_{best_model_name}", scaler, X_train.columns, config
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "model")

        logger.info("Model training completed successfully!")

        return models, results, best_model_name


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Configure matplotlib for headless environment
    import matplotlib

    matplotlib.use("Agg")

    # Train models
    models, results, best_model = train_stellar_models()

    logger.info("Stellar classification training pipeline completed!")
