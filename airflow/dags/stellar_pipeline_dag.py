from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
import logging
import pandas as pd
import importlib.util

# Add src path - handle both local and Docker environments
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from dags folder

# Use environment variable if set, otherwise use detected project root
if "PROJECT_DIR" in os.environ:
    project_root = os.environ["PROJECT_DIR"]
elif "AIRFLOW_HOME" in os.environ:
    project_root = os.environ["AIRFLOW_HOME"]

src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

# Import stellar ingestion functions from standalone module file
stellar_ingestion_path = os.path.join(src_path, "stellar_ingestion.py")
spec = importlib.util.spec_from_file_location(
    "stellar_ingestion_module", stellar_ingestion_path
)
if spec and spec.loader:
    si = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(si)
else:
    raise ImportError(
        f"Could not load stellar_ingestion.py from {stellar_ingestion_path}"
    )

# Import stellar training functions from package
from stellar_ingestion import (
    train_stellar_models,
    train_svm_model,
    train_random_forest_model,
    evaluate_model,
    hyperparameter_tuning,
    save_model_artifacts,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    "stellar_classification_pipeline",
    default_args=default_args,
    description="End-to-end stellar classification ML pipeline",
    schedule="@weekly",
    catchup=False,
    tags=["ml", "stellar", "classification"],
)


def ingest_to_database(**context):
    """
    Ingest CSV data into MariaDB ColumnStore database.

    This task implements the first stage of the database-driven pipeline,
    loading raw data into the 'One Big Table' structure for optimal ML performance.
    """
    logging.info("=== STARTING DATABASE INGESTION ===")

    config = si.load_config()
    data_path = os.path.join(project_root, config.get("data_path", "data/raw/star_classification.csv"))

    try:
        rows_inserted = si.ingest_data_to_db(data_path)
        logging.info(f"Successfully ingested {rows_inserted} rows into database")
        return rows_inserted
    except Exception as e:
        logging.error(f"Database ingestion failed: {e}")
        raise


def preprocess_database_data(**context):
    """
    Process raw data in database and update with engineered features.

    This task performs feature engineering and preprocessing directly
    in the database, updating records in-place.
    """
    logging.info("=== STARTING DATABASE PREPROCESSING ===")

    try:
        rows_processed = si.process_and_update_db()
        logging.info(f"Successfully processed {rows_processed} records in database")
        return rows_processed
    except Exception as e:
        logging.error(f"Database preprocessing failed: {e}")
        raise


def load_training_data_from_db(**context):
    """
    Load processed training data from database for model training.

    This task retrieves the preprocessed data from the database
    and prepares it for the ML training pipeline.
    """
    logging.info("=== LOADING TRAINING DATA FROM DATABASE ===")

    try:
        # Load processed data from database
        df = si.load_data_from_db(processed_only=True)

        if df.empty:
            raise ValueError("No processed data available in database")

        # Save to temporary location for downstream tasks
        temp_path = "data/temp/db_training_data.parquet"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        df.to_parquet(temp_path, index=False)

        logging.info(f"Loaded {len(df)} training records from database")
        return temp_path
    except Exception as e:
        logging.error(f"Loading training data from database failed: {e}")
        raise


def load_data(**context):
    """Load stellar classification data"""
    logging.info("=== STARTING STELLAR DATA LOADING ===")

    config = si.load_config()
    data_path = os.path.join(project_root, config.get("data_path", "data/raw/star_classification.csv"))

    df = si.load_stellar_data(data_path)

    # Save raw data to temp location - use relative path for local execution
    raw_path = os.path.join(project_root, "data/temp/raw_data.parquet")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_parquet(raw_path, index=False)

    logging.info(f"Loaded stellar data with shape: {df.shape}")
    return raw_path


def clean_data(**context):
    """Clean stellar classification data"""
    logging.info("=== STARTING STELLAR DATA CLEANING ===")

    raw_path = context["task_instance"].xcom_pull(task_ids="load_file_data")
    df = pd.read_parquet(raw_path)

    # Load config for cleaning
    config = si.load_config()
    df_clean = si.clean_stellar_data(df, config)

    # Save cleaned data
    clean_path = "data/temp/clean_data.parquet"
    df_clean.to_parquet(clean_path, index=False)

    return clean_path


def exploratory_data_analysis(**context):
    """Perform EDA on stellar data"""
    logging.info("=== STARTING STELLAR EDA ===")

    clean_path = context["task_instance"].xcom_pull(task_ids="clean_file_data")
    df = pd.read_parquet(clean_path)

    # Load config for EDA
    config = si.load_config()
    si.perform_eda(df, config)

    return "EDA completed"


def feature_engineering_task(**context):
    """Perform feature engineering on stellar data"""
    logging.info("=== STARTING STELLAR FEATURE ENGINEERING ===")

    clean_path = context["task_instance"].xcom_pull(task_ids="clean_file_data")
    df = pd.read_parquet(clean_path)

    # Load config for feature engineering
    config = si.load_config()
    df_features = si.feature_engineering(df, config)

    # Save feature engineered data
    features_path = "data/temp/features_data.parquet"
    df_features.to_parquet(features_path, index=False)

    return features_path

    return features_path


def encode_target_task(**context):
    """Encode target variable for stellar classification"""
    logging.info("=== STARTING TARGET ENCODING ===")

    features_path = context["task_instance"].xcom_pull(task_ids="feature_engineering")
    df = pd.read_parquet(features_path)

    # Load config for target encoding
    config = si.load_config()
    df_encoded = si.encode_target_variable(df, config)

    # Save encoded data
    encoded_path = "data/temp/encoded_data.parquet"
    df_encoded.to_parquet(encoded_path, index=False)

    return encoded_path


def split_and_scale(**context):
    """Split data and apply scaling for stellar classification"""
    logging.info("=== STARTING DATA SPLITTING AND SCALING ===")

    encoded_path = context["task_instance"].xcom_pull(task_ids="encode_target")
    df = pd.read_parquet(encoded_path)

    # Load config and perform train-test split and scaling
    config = si.load_config()
    X_train, X_test, y_train, y_test, scaler = si.split_and_scale_data(df, config)

    # Save the split and scaled data
    artifacts = si.save_processed_data(X_train, X_test, y_train, y_test, scaler)

    return artifacts


def train_baseline_models(**context):
    """Train baseline models for stellar classification"""
    import os
    import joblib
    
    logging.info("=== TRAINING BASELINE STELLAR MODELS ===")

    # Set up MLflow tracking
    try:
        import mlflow
        import mlflow.sklearn
        
        # Configure MLflow tracking URI - try environment variable first, then fallback
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://stellar-mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Test MLflow connectivity
        try:
            # Try to connect to MLflow server
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            logging.info(f"✅ MLflow server reachable at {mlflow_uri}")
        except Exception as e:
            logging.warning(f"⚠️ MLflow server not reachable: {e}")
            logging.info("Continuing without MLflow tracking...")
        
        # Set experiment
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'stellar_classification')
        try:
            mlflow.set_experiment(experiment_name)
            logging.info(f"✅ MLflow experiment set: {experiment_name}")
            logging.info(f"✅ MLflow tracking URI: {mlflow_uri}")
        except Exception as e:
            logging.warning(f"⚠️ Could not set MLflow experiment: {e}")
            
    except ImportError:
        logging.warning("⚠️ MLflow not available - proceeding without experiment tracking")

    artifacts_path = context["task_instance"].xcom_pull(task_ids="split_scale_data")

    # Train models with processed data directory
    models, metrics, best_model_name = train_stellar_models(
        "config/datasets/stellar.yaml", "data/processed"
    )

    # Explicitly save the best model to ensure consistent path
    best_model = models[best_model_name]
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/best_{best_model_name}_model.pkl"
    
    logging.info(f"Saving best model ({best_model_name}) to: {best_model_path}")
    joblib.dump(best_model, best_model_path)
    
    # Verify the file was saved
    if os.path.exists(best_model_path):
        logging.info(f"✅ Best model successfully saved to {best_model_path}")
    else:
        logging.error(f"❌ Failed to save model to {best_model_path}")

    # Return only serializable data (exclude model objects and numpy arrays)
    serializable_metrics = {}
    for model_name, model_metrics in metrics.items():
        serializable_metrics[model_name] = {
            "accuracy": model_metrics["accuracy"],
            "classification_report": {
                k: v
                for k, v in model_metrics["classification_report"].items()
                if k not in ["confusion_matrix", "predictions", "probabilities"]
            },
        }

    return {
        "best_model": best_model_name,
        "model_path": best_model_path,  # Pass the exact path to evaluation
        "metrics": serializable_metrics,
        "training_completed": True,
    }


def model_evaluation(**context):
    """Evaluate stellar classification models"""
    import os
    import joblib
    
    logging.info("=== MODEL EVALUATION ===")

    config = si.load_config()

    # Load test data for evaluation
    artifacts_path = context["task_instance"].xcom_pull(task_ids="split_scale_data")

    # Load test data
    X_test = pd.read_csv(artifacts_path["X_test"])
    y_test = pd.read_csv(artifacts_path["y_test"]).iloc[:, 0]  # Get the series

    # Get the best model path from the training task
    train_results = context["task_instance"].xcom_pull(task_ids="train_baseline_models")
    model_path = train_results["model_path"]
    best_model_name = train_results["best_model"]
    
    logging.info(f"Loading best model ({best_model_name}) from: {model_path}")

    # Verify the model file exists before loading
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        # List contents of models directory for debugging
        models_dir = "models"
        if os.path.exists(models_dir):
            available_files = os.listdir(models_dir)
            logging.error(f"Available files in models/: {available_files}")
        raise FileNotFoundError(f"Expected model at {model_path}, but it was not found. Check train_baseline_models to ensure the best model is saved there.")
    
    model = joblib.load(model_path)

    # Evaluate the best model
    evaluation_results = evaluate_model(
        model, X_test, y_test, f"best_{best_model_name}_model", config
    )  # Return only JSON-serializable data (exclude numpy arrays)
    serializable_results = {
        "accuracy": evaluation_results["accuracy"],
        "classification_report": evaluation_results["classification_report"],
        "model_name": f"best_{best_model_name}_model",
        "test_samples": len(y_test),
        "evaluation_completed": True,
    }

    return serializable_results


def save_final_model(**context):
    """Save the final stellar classification model and artifacts"""
    logging.info("=== SAVING FINAL MODEL ===")

    # Get the best model from evaluation
    evaluation_results = context["task_instance"].xcom_pull(task_ids="evaluate_models")

    # Load the best model directly (we know it's random_forest from our simplified pipeline)
    import joblib

    best_model_name = "random_forest"
    model_path = f"models/best_{best_model_name}_model.pkl"
    best_model = joblib.load(model_path)

    # Load feature names and scaler from artifacts
    import json
    import joblib

    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)

    scaler = joblib.load("models/scaler.pkl")
    config = si.load_config()

    # Save model artifacts using the correct function signature
    saved_paths = save_model_artifacts(
        model=best_model,
        model_name="stellar_classifier_final",
        scaler=scaler,
        feature_names=feature_names,
        config=config,
        models_dir="models/",
    )

    logging.info(f"Model saved to: {saved_paths}")
    return saved_paths


# Define tasks

# Database-specific tasks for MariaDB ColumnStore integration
ingest_db_task = PythonOperator(
    task_id="ingest_to_database",
    python_callable=ingest_to_database,
    dag=dag,
)

preprocess_db_task = PythonOperator(
    task_id="preprocess_database_data",
    python_callable=preprocess_database_data,
    dag=dag,
)

load_db_training_task = PythonOperator(
    task_id="load_training_data_from_db",
    python_callable=load_training_data_from_db,
    dag=dag,
)

# Traditional file-based tasks (kept for compatibility)
load_file_task = PythonOperator(
    task_id="load_file_data",
    python_callable=load_data,
    dag=dag,
)

clean_file_task = PythonOperator(
    task_id="clean_file_data",
    python_callable=clean_data,
    dag=dag,
)

eda_task = PythonOperator(
    task_id="exploratory_data_analysis",
    python_callable=exploratory_data_analysis,
    dag=dag,
)

feature_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering_task,
    dag=dag,
)

encode_task = PythonOperator(
    task_id="encode_target",
    python_callable=encode_target_task,
    dag=dag,
)

split_scale_task = PythonOperator(
    task_id="split_scale_data",
    python_callable=split_and_scale,
    dag=dag,
)

train_base_task = PythonOperator(
    task_id="train_baseline_models",
    python_callable=train_baseline_models,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_models",
    python_callable=model_evaluation,
    dag=dag,
)

save_task = PythonOperator(
    task_id="save_final_model",
    python_callable=save_final_model,
    dag=dag,
)

# Define task dependencies

# Database-driven workflow (primary) - Uses MariaDB ColumnStore for scalable data management
ingest_db_task >> preprocess_db_task >> load_db_training_task >> train_base_task

# Alternative file-based workflow (for compatibility/fallback)
(
    load_file_task
    >> clean_file_task
    >> eda_task
    >> feature_task
    >> encode_task
    >> split_scale_task
)

# Common downstream tasks (work with both workflows)
split_scale_task >> train_base_task >> evaluate_task >> save_task
load_db_training_task >> evaluate_task >> save_task
