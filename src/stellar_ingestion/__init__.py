"""
Stellar Classification Package

This package provides data ingestion and training functions for stellar classification.
"""

# Import all ingestion functions
from .ingestion import (
    load_config,
    load_stellar_data,
    clean_stellar_data,
    encode_target_variable,
    detect_and_remove_outliers,
    perform_eda,
    feature_engineering,
    split_and_scale_data,
    save_processed_data,
)

# Import training functions
from .training import (
    train_stellar_models,
    load_training_data,
    train_svm_model,
    train_random_forest_model,
    evaluate_model,
    hyperparameter_tuning,
    save_model_artifacts,
)

__all__ = [
    # Ingestion functions
    "load_config",
    "load_stellar_data",
    "clean_stellar_data",
    "encode_target_variable",
    "detect_and_remove_outliers",
    "perform_eda",
    "feature_engineering",
    "split_and_scale_data",
    "save_processed_data",
    # Training functions
    "train_stellar_models",
    "load_training_data",
    "train_svm_model",
    "train_random_forest_model",
    "evaluate_model",
    "hyperparameter_tuning",
    "save_model_artifacts",
]
