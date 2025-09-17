# 🚀 Portable Setup Guide

This guide helps you set up the Stellar Classification MLOps Pipeline on any Ubuntu system with minimal configuration.

## 📋 Prerequisites

- Ubuntu 18.04+ or compatible Linux distribution
- Conda/Miniconda installed
- Git (for cloning the repository)

## 🛠️ Quick Setup

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd stellar-classification-mlops
```

### 2. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Start the Pipeline
```bash
./start.sh
```

### 4. Access Services
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000/docs

### 5. Stop Services
```bash
./stop.sh
```

## 🔧 What the Setup Does

The setup script automatically:

1. **Creates conda environment** (`stellar-mlops`) with Python 3.9
2. **Installs all dependencies** from requirements.txt
3. **Creates directory structure** for data, models, logs
4. **Configures environment variables** in `.env` file
5. **Initializes Airflow database** and creates admin user
6. **Sets up relative paths** for portability

## 📁 Project Structure

```
stellar-classification-mlops/
├── .env                     # Environment configuration (auto-generated)
├── activate.sh              # Manual environment activation
├── setup.sh                 # One-time setup script
├── start.sh                 # Start all services
├── stop.sh                  # Stop all services
├── airflow/
│   └── dags/               # Airflow DAG definitions
├── api/                    # FastAPI application
├── config/                 # Configuration files
├── data/                   # Data storage (raw, processed, temp)
├── logs/                   # Service logs
├── models/                 # Trained model artifacts
├── src/                    # Source code modules
└── requirements.txt        # Python dependencies
```

## 🌐 Environment Variables

The setup creates a `.env` file with:

- `PROJECT_DIR`: Absolute path to project directory
- `AIRFLOW_HOME`: Airflow configuration directory
- `PYTHONPATH`: Python module search path
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI

## 🔄 Manual Environment Activation

If you need to activate the environment manually:

```bash
source ./activate.sh
```

## 🐛 Troubleshooting

### Port Conflicts
If ports 8080, 5000, or 8000 are in use:
1. Stop conflicting services
2. Or modify ports in `start.sh`

### Permission Issues
```bash
chmod +x *.sh
```

### Conda Not Found
Install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Database Connection Issues
The pipeline works with file-based operations by default. Database features are optional.

## 📊 Running the Pipeline

1. **Access Airflow UI**: http://localhost:8080
2. **Enable the DAG**: `stellar_classification_pipeline`
3. **Trigger manually** or wait for scheduled run
4. **Monitor progress** in Airflow UI
5. **View results** in MLflow UI

## 🔧 Customization

- **Data**: Place your CSV files in `data/raw/`
- **Configuration**: Edit `config/datasets/stellar.yaml`
- **Models**: Modify training parameters in config
- **API**: Extend `api/stellar_main.py` for new endpoints

## 📝 Notes

- All paths are relative to the project directory
- Environment is automatically activated in scripts
- Logs are stored in `logs/` directory
- Models are saved in `models/` directory
- The setup is idempotent (safe to run multiple times)

## 🆘 Support

If you encounter issues:
1. Check log files in `logs/` directory
2. Ensure all prerequisites are installed
3. Verify file permissions on shell scripts
4. Check that required ports are available