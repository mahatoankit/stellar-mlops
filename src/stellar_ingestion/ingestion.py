"""
Stellar Classification Data Ingestion Pipeline

This module provides functions for loading, cleaning, preprocessing, and transforming
stellar classification data for machine learning pipelines.
"""

import os
import logging
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/datasets/stellar.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        else:
            logger.warning(
                f"Config file {config_path} not found. Using default configuration."
            )
            # Return default configuration
            return {
                "data": {
                    "file_path": "data/raw/star_classification.csv",
                    "target_column": "class",
                    "test_size": 0.2,
                    "random_state": 42,
                },
                "preprocessing": {
                    "handle_missing": True,
                    "remove_outliers": True,
                    "scale_features": True,
                },
            }
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        raise


def load_stellar_data(data_path: str) -> pd.DataFrame:
    """
    Load stellar classification dataset.

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame with stellar data
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        logger.info(
            f"✅ Loaded stellar data: {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


def clean_stellar_data(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Clean stellar data by handling missing values.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        Cleaned DataFrame
    """
    try:
        logger.info("🧹 Cleaning stellar data...")
        df_clean = df.copy()

        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        logger.info(f"✅ Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    except Exception as e:
        logger.error(f"❌ Error cleaning data: {e}")
        raise


def encode_target_variable(
    df: pd.DataFrame, config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Encode target variable for classification.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with encoded target
    """
    try:
        logger.info("🏷️ Encoding target variable...")
        df_encoded = df.copy()

        target_col = "class"
        if target_col in df_encoded.columns:
            # Map string classes to numbers
            class_mapping = {"GALAXY": 0, "STAR": 1, "QSO": 2}
            df_encoded[target_col] = df_encoded[target_col].map(class_mapping)
            logger.info(f"Applied class mapping: {class_mapping}")

        return df_encoded
    except Exception as e:
        logger.error(f"❌ Error encoding target: {e}")
        raise


def detect_and_remove_outliers(
    df: pd.DataFrame, config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Detect and remove outliers using Local Outlier Factor.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with outliers removed
    """
    try:
        logger.info("🎯 Detecting and removing outliers...")
        df_clean = df.copy()

        # Select numeric columns (exclude target)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if "class" in numeric_cols:
            numeric_cols.remove("class")

        if len(numeric_cols) > 0:
            # Apply Local Outlier Factor
            lof = LocalOutlierFactor(contamination=0.1)
            outlier_labels = lof.fit_predict(df_clean[numeric_cols])

            # Remove outliers
            df_clean = df_clean[outlier_labels == 1]
            logger.info(f"✅ Removed outliers. New shape: {df_clean.shape}")

        return df_clean
    except Exception as e:
        logger.error(f"❌ Error removing outliers: {e}")
        return df  # Return original data if outlier detection fails


def perform_eda(
    df: pd.DataFrame, config: Dict[str, Any] = None, output_dir: str = "data/plots"
) -> Dict[str, Any]:
    """
    Perform exploratory data analysis.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        output_dir: Directory to save plots

    Returns:
        EDA statistics dictionary
    """
    try:
        logger.info("📊 Performing EDA...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        eda_stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }

        # Class distribution
        if "class" in df.columns:
            class_dist = df["class"].value_counts().to_dict()
            eda_stats["class_distribution"] = class_dist
            logger.info(f"Class distribution: {class_dist}")

        # Generate basic plots if plotting is available
        if PLOTTING_AVAILABLE:
            # Class distribution plot
            if "class" in df.columns:
                plt.figure(figsize=(8, 6))
                df["class"].value_counts().plot(kind="bar")
                plt.title("Class Distribution")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/class_distribution.png")
                plt.close()
                logger.info(f"📈 Saved class distribution plot to {output_dir}")

        logger.info("✅ EDA completed")
        return eda_stats
    except Exception as e:
        logger.error(f"❌ Error during EDA: {e}")
        return {"error": str(e)}


def feature_engineering(
    df: pd.DataFrame, config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Perform feature engineering on stellar data.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with engineered features
    """
    try:
        logger.info("⚙️ Performing feature engineering...")
        df_fe = df.copy()

        # Create color indices (standard astronomical features)
        if all(col in df_fe.columns for col in ["u", "g", "r", "i", "z"]):
            df_fe["u_g"] = df_fe["u"] - df_fe["g"]
            df_fe["g_r"] = df_fe["g"] - df_fe["r"]
            df_fe["r_i"] = df_fe["r"] - df_fe["i"]
            df_fe["i_z"] = df_fe["i"] - df_fe["z"]
            logger.info("✅ Created color indices: u-g, g-r, r-i, i-z")

        logger.info(f"Feature engineering completed. Shape: {df_fe.shape}")
        return df_fe
    except Exception as e:
        logger.error(f"❌ Error in feature engineering: {e}")
        raise


def split_and_scale_data(
    df: pd.DataFrame, config: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split data into train/test and scale features.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        logger.info("🔀 Splitting and scaling data...")

        # Separate features and target
        target_col = "class"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split data
        test_size = config.get("data", {}).get("test_size", 0.2) if config else 0.2
        random_state = config.get("data", {}).get("random_state", 42) if config else 42

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        logger.info(
            f"✅ Data split completed. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}"
        )
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"❌ Error splitting/scaling data: {e}")
        raise


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    output_dir: str = "data/processed",
) -> Dict[str, str]:
    """
    Save processed data and artifacts.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        scaler: Fitted scaler
        output_dir: Output directory

    Returns:
        Dictionary with saved file paths
    """
    try:
        logger.info("💾 Saving processed data...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save datasets
        file_paths = {}
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        file_paths.update(
            {
                "X_train": f"{output_dir}/X_train.csv",
                "X_test": f"{output_dir}/X_test.csv",
                "y_train": f"{output_dir}/y_train.csv",
                "y_test": f"{output_dir}/y_test.csv",
            }
        )

        # Save scaler
        scaler_path = f"{output_dir}/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        file_paths["scaler"] = scaler_path

        # Save feature names
        feature_names_path = f"{output_dir}/feature_names.json"
        import json

        with open(feature_names_path, "w") as f:
            json.dump(list(X_train.columns), f)
        file_paths["feature_names"] = feature_names_path

        logger.info(f"✅ All processed data saved to {output_dir}")
        return file_paths
    except Exception as e:
        logger.error(f"❌ Error saving processed data: {e}")
        raise
