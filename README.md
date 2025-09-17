# 🌟 Stellar Classification MLOps Pipeline

A comprehensive MLOps pipeline for stellar object classification using the SDSS17 dataset. This project demonstrates end-to-end machine learning workflow orchestration with Apache Airflow, MLflow experiment tracking, and FastAPI model serving.

## 🚀 Features

- **Data Pipeline**: Automated data ingestion, cleaning, and preprocessing
- **Model Training**: Support for SVM, Random Forest, and XGBoost models
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Model Serving**: FastAPI endpoints for real-time predictions
- **Orchestration**: Apache Airflow DAGs for workflow automation
- **Containerization**: Docker setup for easy deployment

## 📊 Dataset

This pipeline uses the **SDSS17 Stellar Classification Dataset** containing:

- **100,000 observations** from the Sloan Digital Sky Survey
- **17 features** including photometric measurements (u, g, r, i, z bands)
- **3 classes**: Galaxy, Star, Quasar (QSO)

## 🏗️ Project Structure

```
telco-mlops-pipeline/
├── airflow/
│   └── dags/
│       └── telco_pipeline_dag.py    # Main Airflow DAG
├── src/
│   ├── stellar_ingestion.py         # Data pipeline functions
│   └── stellar_training.py          # Model training functions
├── api/
│   └── stellar_main.py              # FastAPI application
├── config/
│   └── datasets/
│       └── stellar.yaml             # Configuration file
├── data/
│   ├── raw/                         # Raw data files
│   ├── processed/                   # Processed datasets
│   └── temp/                        # Temporary artifacts
├── models/                          # Saved model artifacts
├── Dockerfile                       # Container configuration
├── docker-compose.yml               # Multi-service orchestration
└── requirements.txt                 # Python dependencies
```

## 🛠️ Setup & Installation

### Prerequisites

- Ubuntu 18.04+ or compatible Linux distribution
- Conda/Miniconda installed
- Git (for cloning)

### Quick Start (3 Commands)

```bash
# 1. Clone and navigate
git clone <repository-url>
cd stellar-classification-mlops

# 2. Run automated setup
./setup.sh

# 3. Start all services
./start.sh
```

### Manual Setup (if needed)

```bash
# Create conda environment
conda create -n stellar-mlops python=3.9 -y
conda activate stellar-mlops

# Install dependencies
pip install -r requirements.txt

# Initialize Airflow
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

### Docker Deployment (Alternative)

```bash
docker-compose up -d
```

## 🔄 Pipeline Workflow

### 1. Data Ingestion (`stellar_ingestion.py`)

- **load_stellar_data()**: Load CSV data with validation
- **clean_stellar_data()**: Remove unnecessary features
- **encode_target_variable()**: Encode class labels
- **detect_and_remove_outliers()**: LocalOutlierFactor-based cleaning

### 2. Data Preprocessing

- **perform_eda()**: Generate correlation plots and statistics
- **feature_engineering()**: Create derived features
- **split_and_scale_data()**: Train/test split with StandardScaler
- **SMOTE resampling**: Handle class imbalance

### 3. Model Training (`stellar_training.py`)

- **SVM**: Radial basis function kernel
- **Random Forest**: Ensemble with 100 estimators
- **XGBoost**: Gradient boosting for multiclass

### 4. Model Evaluation

- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- ROC curves (multiclass)
- MLflow experiment logging

### 5. Model Deployment

- FastAPI REST endpoints
- Real-time prediction service
- Model versioning support

## 📈 Monitoring & Tracking

### Airflow UI

- **URL**: http://localhost:8080
- **Credentials**: admin/admin
- **Features**: DAG monitoring, task logs, scheduling

### MLflow UI

- **URL**: http://localhost:5000
- **Features**: Experiment tracking, model registry, metrics visualization

### FastAPI Docs

- **URL**: http://localhost:8000/docs
- **Features**: Interactive API documentation, model prediction endpoints

## 🔧 Configuration

All pipeline parameters are configurable via `config/datasets/stellar.yaml`:

```yaml
dataset:
  name: "stellar_classification"
  source: "Sloan Digital Sky Survey DR17"

preprocessing:
  outlier_detection:
    method: "LocalOutlierFactor"
    threshold: -1.5
  scaling:
    method: "StandardScaler"
  sampling:
    method: "SMOTE"

models:
  svm:
    kernel: "rbf"
    C: 1
  random_forest:
    n_estimators: 100
```

## 🚦 Running the Pipeline

### Start/Stop Services

```bash
# Start all services
./start.sh

# Stop all services  
./stop.sh

# Activate environment manually
source ./activate.sh
```

### Execute Pipeline

```bash
# Trigger DAG from CLI
airflow dags trigger stellar_classification_pipeline

# Or use Airflow UI at http://localhost:8080
```

### Scheduled Execution

- Configure `schedule_interval` in DAG definition
- Default: Weekly execution

## 📝 API Usage

### Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "u": 19.47,
       "g": 17.04,
       "r": 16.00,
       "i": 15.60,
       "z": 15.26,
       "redshift": 0.539
     }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 📊 Model Performance

Expected performance metrics on test data:

- **Overall Accuracy**: ~97%
- **Galaxy Classification**: F1 > 0.98
- **Star Classification**: F1 > 0.96
- **QSO Classification**: F1 > 0.95

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SDSS Collaboration** for the stellar classification dataset
- **Apache Airflow** community for workflow orchestration
- **MLflow** team for experiment tracking capabilities
- **FastAPI** developers for the modern API framework

---

**Made with ❤️ by the MLOps Team**
