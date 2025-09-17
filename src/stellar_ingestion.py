"""
Stellar Classification Data Ingestion Pipeline

This module provides functions for loading, cleaning, preprocessing, and transforming
stellar classification data for machine learning pipelines. Supports both file-based
and MariaDB ColumnStore database operations for scalable MLOps workflows.

Author: MLOps Team
Date: September 2025
"""

print("DEBUG: stellar_ingestion module is being loaded...")

import os
import logging
from pathlib import Path
from collections import Counter
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Database utilities for MariaDB integration
try:
    from db_utils import (
        get_db_connection,
        test_connection,
        execute_query,
        dataframe_to_db,
        query_to_dataframe,
        get_table_info,
    )

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("WARNING: Database utilities not available. Using file-based operations.")

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = sns = None

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
print("DEBUG: After logger configuration")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    try:
        # If no config path provided, try to find it relative to project root
        if config_path is None:
            # Try to find project root from environment or current location
            project_root = os.environ.get('PROJECT_DIR', os.getcwd())
            config_path = os.path.join(project_root, "config/datasets/stellar.yaml")
        
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(
                f"Config file {config_path} not found. Using default configuration."
            )
            # Return default configuration
            return {
                "data": {
                    "target_column": "class",
                    "feature_columns": [
                        "u",
                        "g",
                        "r",
                        "i",
                        "z",
                        "specobjid",
                        "redshift",
                        "plate",
                        "mjd",
                        "fiberid",
                    ],
                    "class_mapping": {"GALAXY": 0, "STAR": 1, "QSO": 2},
                },
                "preprocessing": {
                    "handle_missing": True,
                    "remove_outliers": True,
                    "outlier_method": "lof",
                    "outlier_threshold": 0.1,
                },
                "feature_engineering": {
                    "create_color_indices": True,
                    "scale_features": True,
                },
            }

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"✅ Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        raise


def load_stellar_data(data_path: str) -> pd.DataFrame:
    """
    Load stellar classification dataset.

    Args:
        data_path: Path to the CSV file containing stellar data

    Returns:
        DataFrame with stellar data

    Raises:
        FileNotFoundError: If data file doesn't exist
        pd.errors.EmptyDataError: If data file is empty
    """
    try:
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load the data
        df = pd.read_csv(data_path)

        if df.empty:
            raise pd.errors.EmptyDataError("Data file is empty")

        logger.info(
            f"✅ Loaded stellar data: {df.shape[0]} rows, {df.shape[1]} columns"
        )
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"❌ Error loading data from {data_path}: {e}")
        raise


def ingest_data_to_db(data_path: str) -> int:
    """
    Ingest CSV data into MariaDB ColumnStore database.

    This function implements the "One Big Table" approach where all features
    and metadata are stored together for optimal ML performance.

    Design Rationale:
    - Single table design eliminates JOINs for ML data retrieval
    - MariaDB ColumnStore optimized for analytical queries on large datasets
    - Idempotent operation prevents duplicate data on repeated runs

    Args:
        data_path: Path to the CSV file containing stellar data

    Returns:
        Number of rows inserted into database

    Raises:
        Exception: For database connection or insertion errors
    """
    if not DATABASE_AVAILABLE:
        raise RuntimeError(
            "Database functionality not available. Check db_utils import."
        )

    try:
        logger.info(f"🚀 Starting data ingestion from {data_path} to MariaDB...")

        # Test database connection
        if not test_connection():
            raise RuntimeError("Cannot connect to MariaDB database")

        # Load data from CSV
        df = load_stellar_data(data_path)

        # Prepare data for database insertion
        # Rename 'delta' to 'delta_coord' to avoid SQL reserved keyword
        if "delta" in df.columns:
            df = df.rename(columns={"delta": "delta_coord"})

        # Add processing metadata
        df["is_processed"] = False
        df["data_split"] = None

        # Check if data already exists (idempotent operation)
        existing_count_query = "SELECT COUNT(*) as count FROM stellar_data"
        existing_result = execute_query(existing_count_query, fetch=True)
        existing_count = existing_result[0]["count"] if existing_result else 0

        if existing_count > 0:
            logger.warning(
                f"Database already contains {existing_count} records. Clearing for fresh ingestion."
            )
            # Clear existing data for fresh ingestion
            clear_query = "TRUNCATE TABLE stellar_data"
            execute_query(clear_query)
            logger.info("Cleared existing data from stellar_data table")

        # Limit data size for testing (first 1000 rows)
        if len(df) > 1000:
            logger.info(f"Limiting dataset from {len(df)} to 1000 rows for testing")
            df = df.head(1000)

        # Insert data into database using ColumnStore optimized bulk loading
        rows_inserted = dataframe_to_db(df, "stellar_data", if_exists="append")

        # Log pipeline run
        run_metadata = {
            "source_file": data_path,
            "rows_processed": rows_inserted,
            "columns": list(df.columns),
        }

        insert_run_query = """
        INSERT INTO pipeline_runs (run_id, run_type, status, records_processed, metadata) 
        VALUES (%s, %s, %s, %s, %s)
        """
        import json

        execute_query(
            insert_run_query,
            (
                f"INGEST_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                "INGESTION",
                "COMPLETED",
                rows_inserted,
                json.dumps(run_metadata),
            ),
        )

        logger.info(
            f"✅ Successfully ingested {rows_inserted} rows into stellar_data table"
        )
        return rows_inserted

    except Exception as e:
        logger.error(f"❌ Error during data ingestion: {e}")

        # Log failed run
        if DATABASE_AVAILABLE:
            try:
                insert_run_query = """
                INSERT INTO pipeline_runs (run_id, run_type, status, error_message) 
                VALUES (%s, %s, %s, %s)
                """
                execute_query(
                    insert_run_query,
                    (
                        f"INGEST_FAIL_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        "INGESTION",
                        "FAILED",
                        str(e),
                    ),
                )
            except:
                pass  # Don't fail on logging errors

        raise


def load_data_from_db(processed_only: bool = False) -> pd.DataFrame:
    """
    Load stellar data from MariaDB database.

    Args:
        processed_only: If True, only return preprocessed data

    Returns:
        DataFrame with stellar data from database

    Raises:
        Exception: For database connection or query errors
    """
    if not DATABASE_AVAILABLE:
        raise RuntimeError(
            "Database functionality not available. Check db_utils import."
        )

    try:
        logger.info(
            f"📊 Loading data from database (processed_only={processed_only})..."
        )

        if processed_only:
            query = "SELECT * FROM training_data_view"
        else:
            query = "SELECT * FROM stellar_data ORDER BY id"

        df = query_to_dataframe(query)

        logger.info(f"✅ Loaded {len(df)} rows from database")
        return df

    except Exception as e:
        logger.error(f"❌ Error loading data from database: {e}")
        raise


def clean_stellar_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Clean stellar data by handling missing values and data types.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        Cleaned DataFrame
    """
    try:
        logger.info("🧹 Starting data cleaning...")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Log initial data info
        logger.info(f"Initial shape: {df_clean.shape}")
        logger.info(f"Missing values: {df_clean.isnull().sum().sum()}")

        # Handle missing values
        if config.get("preprocessing", {}).get("handle_missing", True):
            # For numerical columns, fill with median
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    logger.info(f"Filled missing values in {col} with median")

        # Remove rows with missing target variable
        target_col = config.get("data", {}).get("target_column", "class")
        if target_col in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[target_col])
            dropped_rows = initial_rows - len(df_clean)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing target variable")

        logger.info(f"✅ Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean

    except Exception as e:
        logger.error(f"❌ Error during data cleaning: {e}")
        raise


def encode_target_variable(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Encode target variable for classification.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with encoded target variable
    """
    try:
        logger.info("🏷️ Encoding target variable...")

        df_encoded = df.copy()
        target_col = config.get("data", {}).get("target_column", "class")

        if target_col not in df_encoded.columns:
            logger.warning(f"Target column '{target_col}' not found in data")
            return df_encoded

        # Get class mapping
        class_mapping = config.get("data", {}).get("class_mapping", {})

        if class_mapping:
            # Use provided mapping
            df_encoded[target_col] = df_encoded[target_col].map(class_mapping)
            logger.info(f"Applied class mapping: {class_mapping}")
        else:
            # Create automatic mapping
            unique_classes = df_encoded[target_col].unique()
            class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
            df_encoded[target_col] = df_encoded[target_col].map(class_mapping)
            logger.info(f"Created automatic class mapping: {class_mapping}")

        # Log class distribution
        class_dist = df_encoded[target_col].value_counts().to_dict()
        logger.info(f"Class distribution: {class_dist}")

        return df_encoded

    except Exception as e:
        logger.error(f"❌ Error encoding target variable: {e}")
        raise


def detect_and_remove_outliers(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Detect and remove outliers using specified method.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with outliers removed
    """
    try:
        if not config.get("preprocessing", {}).get("remove_outliers", True):
            logger.info("Outlier removal is disabled in config")
            return df

        logger.info("🎯 Detecting and removing outliers...")

        df_clean = df.copy()
        initial_rows = len(df_clean)

        # Select numerical columns for outlier detection (exclude target)
        target_col = config.get("data", {}).get("target_column", "class")
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_columns:
            numeric_columns.remove(target_col)

        if not numeric_columns:
            logger.warning("No numerical columns found for outlier detection")
            return df_clean

        # Apply outlier detection
        outlier_method = config.get("preprocessing", {}).get("outlier_method", "lof")
        outlier_threshold = config.get("preprocessing", {}).get(
            "outlier_threshold", 0.1
        )

        if outlier_method.lower() == "lof":
            # Local Outlier Factor
            lof = LocalOutlierFactor(contamination=outlier_threshold, n_neighbors=20)
            outlier_labels = lof.fit_predict(df_clean[numeric_columns])

            # Remove outliers (LOF returns -1 for outliers, 1 for normal points)
            df_clean = df_clean[outlier_labels == 1]

        removed_rows = initial_rows - len(df_clean)
        logger.info(
            f"✅ Removed {removed_rows} outliers ({removed_rows/initial_rows*100:.2f}%)"
        )

        return df_clean

    except Exception as e:
        logger.error(f"❌ Error during outlier detection: {e}")
        return df  # Return original data if outlier detection fails


def perform_eda(
    df: pd.DataFrame, config: Dict[str, Any], output_dir: str = "data/plots"
) -> Dict[str, Any]:
    """
    Perform exploratory data analysis and generate plots.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        output_dir: Directory to save plots

    Returns:
        Dictionary with EDA statistics
    """
    try:
        logger.info("📊 Performing exploratory data analysis...")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        eda_stats = {}

        # Basic statistics
        eda_stats["shape"] = df.shape
        eda_stats["columns"] = list(df.columns)
        eda_stats["dtypes"] = df.dtypes.to_dict()
        eda_stats["missing_values"] = df.isnull().sum().to_dict()
        eda_stats["numeric_summary"] = df.describe().to_dict()

        # Class distribution
        target_col = config.get("data", {}).get("target_column", "class")
        if target_col in df.columns:
            class_dist = df[target_col].value_counts().to_dict()
            eda_stats["class_distribution"] = class_dist
            logger.info(f"Class distribution: {class_dist}")

        # Generate plots if plotting is available
        if PLOTTING_AVAILABLE:
            plt.style.use("default")

            # Class distribution plot
            if target_col in df.columns:
                plt.figure(figsize=(8, 6))
                df[target_col].value_counts().plot(kind="bar")
                plt.title("Class Distribution")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/class_distribution.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

            # Correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 10))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
                plt.title("Feature Correlation Matrix")
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

                eda_stats["correlation_matrix"] = correlation_matrix.to_dict()

            logger.info(f"📈 EDA plots saved to {output_dir}")
        else:
            logger.warning(
                "Plotting libraries not available. Skipping plot generation."
            )

        logger.info("✅ EDA completed")
        return eda_stats

    except Exception as e:
        logger.error(f"❌ Error during EDA: {e}")
        return {"error": str(e)}

        raise


def process_and_update_db() -> int:
    """
    Process raw data in database and update with engineered features.

    This function performs feature engineering and preprocessing directly
    in the database, following the "One Big Table" approach for optimal
    ML performance.

    Design Rationale:
    - Updates records in-place rather than creating separate tables
    - Maintains data lineage with processing flags
    - Optimized for ColumnStore analytical queries

    Returns:
        Number of records processed and updated

    Raises:
        Exception: For database operations or processing errors
    """
    if not DATABASE_AVAILABLE:
        raise RuntimeError(
            "Database functionality not available. Check db_utils import."
        )

    try:
        logger.info("🔄 Starting database preprocessing and feature engineering...")

        # Load unprocessed data from database
        unprocessed_query = "SELECT * FROM stellar_data WHERE is_processed = FALSE"
        df = query_to_dataframe(unprocessed_query)

        if df.empty:
            logger.info("No unprocessed data found in database.")
            return 0

        logger.info(f"Processing {len(df)} unprocessed records...")

        # Load configuration
        config = load_config()

        # Perform cleaning and feature engineering
        df_clean = clean_stellar_data(df, config)
        df_engineered = feature_engineering(df_clean, config)

        # Prepare update queries for each record
        update_queries = []
        for _, row in df_engineered.iterrows():
            # Calculate color indices - use the column names from feature engineering
            u_g_color = row.get("u_g", row.get("u", 0) - row.get("g", 0))
            g_r_color = row.get("g_r", row.get("g", 0) - row.get("r", 0))
            r_i_color = row.get("r_i", row.get("r", 0) - row.get("i", 0))
            i_z_color = row.get("i_z", row.get("i", 0) - row.get("z", 0))

            update_query = """
            UPDATE stellar_data 
            SET u_g_color = %s, g_r_color = %s, r_i_color = %s, i_z_color = %s,
                is_processed = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """

            update_queries.append(
                (u_g_color, g_r_color, r_i_color, i_z_color, row["id"])
            )

        # Execute bulk update
        if update_queries:
            for query_params in update_queries:
                execute_query(
                    """UPDATE stellar_data 
                       SET u_g_color = %s, g_r_color = %s, r_i_color = %s, i_z_color = %s,
                           is_processed = TRUE, updated_at = CURRENT_TIMESTAMP
                       WHERE id = %s""",
                    query_params,
                )

        # Log pipeline run
        run_metadata = {
            "processed_records": len(df),
            "feature_columns": ["u_g_color", "g_r_color", "r_i_color", "i_z_color"],
        }

        insert_run_query = """
        INSERT INTO pipeline_runs (run_id, run_type, status, records_processed, metadata) 
        VALUES (%s, %s, %s, %s, %s)
        """
        import json

        execute_query(
            insert_run_query,
            (
                f"PREPROCESS_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                "PREPROCESSING",
                "COMPLETED",
                len(df),
                json.dumps(run_metadata),
            ),
        )

        logger.info(
            f"✅ Successfully processed and updated {len(df)} records in database"
        )
        return len(df)

    except Exception as e:
        logger.error(f"❌ Error during database preprocessing: {e}")

        # Log failed run
        if DATABASE_AVAILABLE:
            try:
                insert_run_query = """
                INSERT INTO pipeline_runs (run_id, run_type, status, error_message) 
                VALUES (%s, %s, %s, %s)
                """
                execute_query(
                    insert_run_query,
                    (
                        f"PREPROCESS_FAIL_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        "PREPROCESSING",
                        "FAILED",
                        str(e),
                    ),
                )
            except:
                pass  # Don't fail on logging errors

        raise


def feature_engineering(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
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

        # Create color indices if configured
        if config.get("feature_engineering", {}).get("create_color_indices", True):
            # Standard astronomical color indices
            if all(col in df_fe.columns for col in ["u", "g", "r", "i", "z"]):
                df_fe["u_g"] = df_fe["u"] - df_fe["g"]
                df_fe["g_r"] = df_fe["g"] - df_fe["r"]
                df_fe["r_i"] = df_fe["r"] - df_fe["i"]
                df_fe["i_z"] = df_fe["i"] - df_fe["z"]
                logger.info("✅ Created color indices: u-g, g-r, r-i, i-z")

        logger.info(f"Feature engineering completed. Final shape: {df_fe.shape}")
        return df_fe

    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
        raise


def split_and_scale_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split data into train/test sets and scale features.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        logger.info("🔀 Splitting and scaling data...")

        # Separate features and target
        target_col = config.get("data", {}).get("target_column", "class")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split the data
        test_size = config.get("data", {}).get("test_size", 0.2)
        random_state = config.get("data", {}).get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Train set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")

        # Scale features if configured
        scaler = None
        if config.get("feature_engineering", {}).get("scale_features", True):
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns, index=X_test.index
            )
            logger.info("✅ Features scaled using StandardScaler")
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        else:
            logger.info("Feature scaling disabled")
            return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        logger.error(f"❌ Error during data splitting and scaling: {e}")
        raise


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: Optional[StandardScaler],
    output_dir: str = "data/processed",
) -> Dict[str, str]:
    """
    Save processed data and artifacts.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        scaler: Fitted scaler object
        output_dir: Output directory

    Returns:
        Dictionary with file paths
    """
    try:
        logger.info("💾 Saving processed data...")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_paths = {}

        # Save datasets
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
        if scaler is not None:
            scaler_path = f"{output_dir}/scaler.pkl"
            joblib.dump(scaler, scaler_path)
            file_paths["scaler"] = scaler_path
            logger.info(f"Scaler saved to {scaler_path}")

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


if __name__ == "__main__":
    # Test the functions
    print("Testing stellar ingestion functions...")

    try:
        # Test configuration loading
        config = load_config()
        print("✅ Config loading test passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
