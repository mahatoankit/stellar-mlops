# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - AIRFLOW CONTAINER
# =============================================================================

FROM apache/airflow:2.7.3-python3.9

# Switch to root for system packages
USER root

# Install system dependencies for data science packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libfontconfig1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p \
    /opt/airflow/logs \
    /opt/airflow/logs/scheduler \
    /opt/airflow/logs/dag_processor_manager \
    /opt/airflow/dags \
    /opt/airflow/plugins \
    /opt/airflow/data \
    /opt/airflow/data/temp \
    /opt/airflow/data/processed \
    /opt/airflow/models \
    /opt/airflow/src \
    /opt/airflow/config \
    && chown -R airflow:root /opt/airflow \
    && chmod -R 775 /opt/airflow

# Switch back to airflow user
USER airflow

# Install Python packages
RUN pip install --no-cache-dir \
    # Database Connectivity
    mysql-connector-python==8.0.33 \
    PyMySQL==1.1.0 \
    SQLAlchemy==1.4.53 \
    # Data Processing
    pandas==2.1.4 \
    numpy==1.26.4 \
    pyarrow==14.0.2 \
    pyyaml==6.0.1 \
    # Machine Learning
    scikit-learn==1.3.2 \
    imbalanced-learn==0.11.0 \
    xgboost==1.7.6 \
    # Visualization
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    # MLflow
    mlflow==2.8.1 \
    # Utilities
    joblib==1.3.1

# Copy project files
COPY --chown=airflow:root ./airflow/dags /opt/airflow/dags
COPY --chown=airflow:root ./src /opt/airflow/src
COPY --chown=airflow:root ./config /opt/airflow/config

# Set environment variables
ENV PYTHONPATH="/opt/airflow/src:$PYTHONPATH"
ENV AIRFLOW_HOME="/opt/airflow"

# Expose Airflow webserver port
EXPOSE 8080
