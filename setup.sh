#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - ENVIRONMENT SETUP
# =============================================================================

set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🌟 Setting up Stellar Classification MLOps Pipeline..."
echo "📁 Project directory: $PROJECT_DIR"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment (if it doesn't exist)
if ! conda env list | grep -q "stellar-mlops"; then
    echo "📦 Creating conda environment..."
    conda create -n stellar-mlops python=3.9 -y
else
    echo "✅ Conda environment 'stellar-mlops' already exists"
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stellar-mlops

# Install requirements
echo "⬇️ Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,temp,plots}
mkdir -p models
mkdir -p airflow/{dags,logs,plugins}
mkdir -p mlflow_artifacts
mkdir -p logs/{scheduler,webserver}

# Create environment configuration file
echo "🔧 Creating environment configuration..."
cat > .env << EOF
# Stellar Classification MLOps Pipeline Environment
PROJECT_DIR=$PROJECT_DIR
AIRFLOW_HOME=$PROJECT_DIR
PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH
AIRFLOW__CORE__DAGS_FOLDER=$PROJECT_DIR/airflow/dags
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
AIRFLOW__CORE__XCOM_BACKEND=airflow.models.xcom.BaseXCom
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__AUTH_MANAGER=airflow.auth.managers.fab.fab_auth_manager.FabAuthManager
MLFLOW_TRACKING_URI=file://$PROJECT_DIR/mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=$PROJECT_DIR/mlflow_artifacts
EOF

# Source environment variables
source .env

# Initialize Airflow database (if not already done)
if [ ! -f "airflow.db" ]; then
    echo "🗄️ Initializing Airflow database..."
    airflow db init
else
    echo "✅ Airflow database already initialized"
fi

# Create admin user (if not exists)
echo "👤 Creating admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com 2>/dev/null || echo "ℹ️ Admin user already exists"

# Set permissions
chmod +x start.sh stop.sh

# Create a simple activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the stellar-mlops environment and set variables
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stellar-mlops
source "$PROJECT_DIR/.env"
echo "✅ Environment activated. Project directory: $PROJECT_DIR"
EOF
chmod +x activate.sh

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the pipeline:"
echo "   ./start.sh"
echo ""
echo "🔄 To activate environment manually:"
echo "   source ./activate.sh"
echo ""
echo "🛑 To stop all services:"
echo "   ./stop.sh"
echo ""
echo "🌐 Access points (after starting):"
echo "   - Airflow UI: http://localhost:8080 (admin/admin)"
echo "   - MLflow UI: http://localhost:5000"
echo "   - FastAPI: http://localhost:8000/docs"