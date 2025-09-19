"""
Stellar Classification API
==========================

FastAPI-based REST API for stellar classification inference using trained ML models.
Provides endpoints for model prediction, health checks, and system diagnostics.

Features:
- Real-time stellar classification predictions
- Model health monitoring and diagnostics
- Database connectivity status
- Comprehensive error handling and logging
- OpenAPI/Swagger documentation

Author: MLOPS Team
Version: 1.0.0
Framework: FastAPI, scikit-learn, MLflow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import json
from typing import Dict, Any, List

# ============================================================================
# Environment Configuration
# ============================================================================

# Add src path for database utilities (relative to current file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

# ============================================================================
# Database Integration
# ============================================================================

# Database utilities for MariaDB integration
try:
    from db_utils import (
        get_db_connection,
        test_connection,
        execute_query,
        query_to_dataframe,
        get_table_info,
    )

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print(
        "WARNING: Database utilities not available. Using file-based operations only."
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stellar Classification API",
    description="API for classifying stellar objects (Galaxy, Star, QSO) using trained ML models",
    version="1.0.0",
)


# Pydantic models for request/response
class StellarData(BaseModel):
    u: float  # u magnitude
    g: float  # g magnitude
    r: float  # r magnitude
    i: float  # i magnitude
    z: float  # z magnitude
    specobjid: float
    redshift: float
    plate: int
    mjd: int
    fiberid: int


class ClassificationResponse(BaseModel):
    predicted_class: str
    prediction_probability: float
    class_probabilities: Dict[str, float]
    confidence: str
    model_version: str


# Global variables for model and scaler
model = None
scaler = None
feature_names = None
class_labels = ["GALAXY", "STAR", "QSO"]
model_info = None


def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, feature_names, model_info

    try:
        # Try multiple possible model directories (relative to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_dirs = [
            os.path.join(project_root, "models"),
            "./models",
            "../models",
        ]

        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                model_dir = dir_path
                break

        if not model_dir:
            logger.warning(
                "No models directory found. API will return dummy responses."
            )
            return False

        # Find the best model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]
        if not model_files:
            logger.warning(f"No model files found in {model_dir}")
            return False

        model_path = os.path.join(model_dir, model_files[0])
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")

        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded successfully")

        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)
            logger.info(f"✅ Feature names loaded: {len(feature_names)} features")

        # Load model info
        model_info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            logger.info("✅ Model info loaded successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Error loading model artifacts: {e}")
        return False


def preprocess_stellar_data(data: StellarData) -> np.ndarray:
    """Preprocess stellar data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # Add color indices (feature engineering)
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z"]

        # Ensure all expected features are present
        if feature_names:
            # Reorder columns to match training data
            df = df.reindex(columns=feature_names, fill_value=0)

        # Scale the data
        if scaler:
            processed_data = scaler.transform(df)
        else:
            processed_data = df.values

        return processed_data

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Loading model artifacts...")
    success = load_model_artifacts()
    if not success:
        logger.warning(
            "Failed to load model artifacts. API will return dummy responses."
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Stellar Classification API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": pd.Timestamp.now().isoformat(),
    }


@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model_info:
        return model_info
    else:
        return {
            "model_name": "stellar_classifier",
            "model_type": type(model).__name__ if model else "Unknown",
            "classes": class_labels,
            "status": "Model info not available",
        }


@app.post("/predict", response_model=ClassificationResponse)
async def predict_stellar_class(stellar_object: StellarData):
    """Predict stellar object class"""
    try:
        if model is None:
            # Return dummy response if model not loaded
            return ClassificationResponse(
                predicted_class="STAR",
                prediction_probability=0.95,
                class_probabilities={"GALAXY": 0.02, "STAR": 0.95, "QSO": 0.03},
                confidence="High",
                model_version="dummy_v1.0",
            )

        # Preprocess data
        processed_data = preprocess_stellar_data(stellar_object)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]

        # Get predicted class name
        predicted_class = class_labels[prediction]

        # Get class probabilities
        class_probs = {
            class_labels[i]: float(prob) for i, prob in enumerate(prediction_proba)
        }

        # Determine confidence
        max_prob = float(np.max(prediction_proba))
        if max_prob > 0.8:
            confidence = "High"
        elif max_prob > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        return ClassificationResponse(
            predicted_class=predicted_class,
            prediction_probability=max_prob,
            class_probabilities=class_probs,
            confidence=confidence,
            model_version=(
                model_info.get("model_name", "v1.0") if model_info else "v1.0"
            ),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch")
async def predict_batch_stellar_objects(stellar_objects: List[StellarData]):
    """Predict classes for multiple stellar objects"""
    try:
        if len(stellar_objects) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 100 objects allowed.",
            )

        results = []
        for stellar_obj in stellar_objects:
            result = await predict_stellar_class(stellar_obj)
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.get("/classes")
async def get_classes():
    """Get available stellar classes"""
    return {
        "classes": class_labels,
        "descriptions": {
            "GALAXY": "Extended extragalactic object",
            "STAR": "Point-like stellar object",
            "QSO": "Quasi-stellar object (quasar)",
        },
    }


@app.get("/features")
async def get_features():
    """Get expected input features"""
    return {
        "required_features": [
            {"name": "u", "description": "u-band magnitude"},
            {"name": "g", "description": "g-band magnitude"},
            {"name": "r", "description": "r-band magnitude"},
            {"name": "i", "description": "i-band magnitude"},
            {"name": "z", "description": "z-band magnitude"},
            {"name": "specobjid", "description": "Spectroscopic object ID"},
            {"name": "redshift", "description": "Redshift value"},
            {"name": "plate", "description": "Plate number"},
            {"name": "mjd", "description": "Modified Julian Date"},
            {"name": "fiberid", "description": "Fiber ID"},
        ],
        "derived_features": [
            "u_g (u-g color index)",
            "g_r (g-r color index)",
            "r_i (r-i color index)",
            "i_z (i-z color index)",
        ],
    }


@app.get("/database/status")
async def get_database_status():
    """
    Check database connection status and get basic statistics.

    This endpoint demonstrates the capability of interacting with the
    MariaDB ColumnStore database for data management operations.
    """
    if not DATABASE_AVAILABLE:
        return {
            "database_available": False,
            "message": "Database functionality not configured",
        }

    try:
        # Test connection
        connection_status = test_connection()

        if not connection_status:
            return {
                "database_available": False,
                "connection_status": "failed",
                "message": "Cannot connect to database",
            }

        # Get table information
        table_info = get_table_info("stellar_data")

        # Get processing statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN is_processed = TRUE THEN 1 ELSE 0 END) as processed_records,
            COUNT(DISTINCT class) as unique_classes,
            COUNT(DISTINCT data_split) as data_splits
        FROM stellar_data
        """
        stats_result = execute_query(stats_query, fetch=True)
        stats = stats_result[0] if stats_result else {}

        return {
            "database_available": True,
            "connection_status": "connected",
            "table_info": table_info,
            "statistics": stats,
            "message": "Database is healthy and accessible",
        }

    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {
            "database_available": False,
            "connection_status": "error",
            "error": str(e),
            "message": "Database error occurred",
        }


@app.get("/database/sample-data")
async def get_sample_data_from_db(limit: int = 10):
    """
    Retrieve sample data from the database.

    This endpoint demonstrates querying the MariaDB ColumnStore database
    and showcases the "One Big Table" approach for ML data access.

    Args:
        limit: Number of sample records to return (default: 10, max: 100)
    """
    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Database functionality not available"
        )

    # Validate limit parameter
    if limit > 100:
        limit = 100
    elif limit < 1:
        limit = 10

    try:
        # Query sample data from database
        sample_query = f"""
        SELECT 
            obj_ID, alpha, delta_coord, u, g, r, i, z,
            u_g_color, g_r_color, r_i_color, i_z_color,
            redshift, class, is_processed, data_split
        FROM stellar_data 
        WHERE is_processed = TRUE
        LIMIT {limit}
        """

        df = query_to_dataframe(sample_query)

        if df.empty:
            return {
                "message": "No processed data available in database",
                "records": [],
                "count": 0,
            }

        # Convert to dictionary format
        records = df.to_dict("records")

        return {
            "message": f"Retrieved {len(records)} sample records from database",
            "records": records,
            "count": len(records),
            "features_included": list(df.columns),
        }

    except Exception as e:
        logger.error(f"Sample data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
