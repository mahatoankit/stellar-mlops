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

    # Add database connectivity test before attempting ingestion
    def test_db_connection():
        """Test database connectivity with detailed error reporting"""
        try:
            import mysql.connector
            from mysql.connector import Error
            
            db_config = {
                'host': os.environ.get('MARIADB_HOST', 'mariadb-columnstore'),
                'port': int(os.environ.get('MARIADB_PORT', 3306)),
                'user': os.environ.get('MARIADB_USER', 'stellar_user'),
                'password': os.environ.get('MARIADB_PASSWORD', 'stellar_user_password'),
                'database': os.environ.get('MARIADB_DATABASE', 'stellar_db')
            }
            
            logging.info(f"Testing connection to {db_config['host']}:{db_config['port']}")
            
            connection = mysql.connector.connect(**db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                connection.close()
                logging.info("✅ Database connection test successful")
                return True
                
        except Error as e:
            logging.error(f"❌ Database connection failed: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Database connection error: {e}")
            return False

    # Add connection retry logic for cross-system compatibility
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Database connection attempt {attempt + 1}/{max_retries}")
            
            # Test database connectivity first
            import time
            if attempt > 0:
                logging.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            
            # Test connection before attempting ingestion
            if not test_db_connection():
                raise Exception("Database connectivity test failed")
            
            rows_inserted = si.ingest_data_to_db(data_path)
            logging.info(f"Successfully ingested {rows_inserted} rows into database")
            return rows_inserted
            
        except Exception as e:
            logging.error(f"Database ingestion attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error("All database connection attempts failed. Checking MariaDB service...")
                # Try to provide helpful error diagnosis
                try:
                    import subprocess
                    result = subprocess.run(['docker', 'compose', 'ps', 'stellar-mariadb'], 
                                          capture_output=True, text=True, cwd=project_root)
                    logging.info(f"MariaDB container status: {result.stdout}")
                except:
                    pass
                raise Exception(f"Database ingestion failed after {max_retries} attempts. Last error: {e}")
            else:
                logging.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue
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
    """Train Random Forest model for stellar classification - Simplified for MLOps Pipeline Demo"""
    import joblib
    
    logging.info("=== TRAINING STELLAR CLASSIFICATION MODEL (Random Forest) ===")

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

    # Simplified: Train only Random Forest model for MLOps demo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    
    # Load training data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv") 
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    logging.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Train Random Forest model
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Random Forest Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))

    # Save the model with consistent naming
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_random_forest_model.pkl"
    
    logging.info(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    
    # Verify the file was saved
    if os.path.exists(model_path):
        logging.info(f"✅ Model successfully saved to {model_path}")
    else:
        logging.error(f"❌ Failed to save model to {model_path}")

    # Log to MLflow if available
    try:
        with mlflow.start_run(run_name="random_forest_training"):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            logging.info("✅ Metrics logged to MLflow")
    except Exception as e:
        logging.warning(f"⚠️ MLflow logging failed: {e}")

    # Return simplified results for downstream tasks - Always Random Forest for MLOps demo
    return {
        "best_model": "random_forest",
        "model_path": model_path,
        "metrics": {
            "random_forest": {
                "accuracy": accuracy,
                "classification_report": {"accuracy": accuracy}
            }
        },
        "training_completed": True,
    }


def model_evaluation(**context):
    """Evaluate stellar classification models"""
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
    """Save the final stellar classification model and artifacts - Simplified for MLOps Demo"""
    logging.info("=== SAVING FINAL MODEL ===")

    # Simplified: Always use Random Forest for consistency
    model_path = "models/best_random_forest_model.pkl"
    
    logging.info(f"Loading Random Forest model from: {model_path}")

    # Load the model
    import joblib

    # Verify the model file exists before loading
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        # List contents of models directory for debugging
        models_dir = "models"
        if os.path.exists(models_dir):
            available_files = os.listdir(models_dir)
            logging.error(f"Available files in models/: {available_files}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    best_model = joblib.load(model_path)

    # Load feature names and scaler from artifacts
    import json

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

    logging.info(f"Final model saved to: {saved_paths}")
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
