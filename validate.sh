#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - VALIDATION SCRIPT
# =============================================================================

set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🔍 Validating Stellar Classification MLOps Pipeline Setup..."
echo "📁 Project directory: $PROJECT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please run ./setup.sh first."
    exit 1
fi

# Load environment variables
source .env

# Check conda environment
if ! conda env list | grep -q "stellar-mlops"; then
    echo "❌ Conda environment 'stellar-mlops' not found."
    exit 1
fi

echo "✅ Conda environment found"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stellar-mlops

# Check required directories
REQUIRED_DIRS=(
    "data/raw"
    "data/processed" 
    "data/temp"
    "data/plots"
    "models"
    "airflow/dags"
    "logs/scheduler"
    "logs/webserver"
    "mlflow_artifacts"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Required directory missing: $dir"
        exit 1
    fi
done

echo "✅ All required directories exist"

# Check required files
REQUIRED_FILES=(
    "requirements.txt"
    "config/datasets/stellar.yaml"
    "airflow/dags/stellar_pipeline_dag.py"
    "api/stellar_main.py"
    "src/stellar_ingestion.py"
    "setup.sh"
    "start.sh"
    "stop.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files exist"

# Check Python packages
echo "🐍 Checking Python packages..."
python -c "
import sys
required_packages = [
    'airflow', 'pandas', 'numpy', 'scikit-learn', 
    'mlflow', 'fastapi', 'uvicorn', 'matplotlib', 'seaborn'
]
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All required packages installed')
"

# Check Airflow database
if [ ! -f "airflow.db" ]; then
    echo "❌ Airflow database not initialized"
    exit 1
fi

echo "✅ Airflow database initialized"

# Test configuration loading
echo "🔧 Testing configuration loading..."
python -c "
import sys, os
sys.path.append('src')
from stellar_ingestion import load_config
try:
    config = load_config()
    print('✅ Configuration loaded successfully')
except Exception as e:
    print(f'❌ Configuration loading failed: {e}')
    sys.exit(1)
"

# Check port availability
echo "🌐 Checking port availability..."
for port in 8080 5000 8000; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is in use (this is OK if services are running)"
    else
        echo "✅ Port $port is available"
    fi
done

# Test import of main modules
echo "📦 Testing module imports..."
python -c "
import sys, os
sys.path.append('src')
try:
    import stellar_ingestion
    print('✅ stellar_ingestion module imported successfully')
except Exception as e:
    print(f'❌ Failed to import stellar_ingestion: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Validation completed successfully!"
echo ""
echo "📋 Summary:"
echo "   ✅ Environment configured"
echo "   ✅ Dependencies installed"
echo "   ✅ Directory structure created"
echo "   ✅ Configuration files present"
echo "   ✅ Airflow initialized"
echo "   ✅ Modules importable"
echo ""
echo "🚀 Ready to start the pipeline with: ./start.sh"