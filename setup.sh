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

# Install system dependencies for PostgreSQL (if not already installed)
echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL service
echo "🐘 Setting up PostgreSQL..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

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

# Install PostgreSQL Python driver
echo "📊 Installing PostgreSQL driver..."
pip install psycopg2-binary

# Set up PostgreSQL database for Airflow
echo "🗄️ Setting up Airflow database..."
sudo -u postgres psql -c "CREATE DATABASE airflow_db;" 2>/dev/null || echo "Database airflow_db already exists"
sudo -u postgres psql -c "CREATE USER admin WITH PASSWORD 'admin';" 2>/dev/null || echo "User admin already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE airflow_db TO admin;"
sudo -u postgres psql -c "ALTER USER admin CREATEDB;"
sudo -u postgres psql -c "ALTER USER admin WITH SUPERUSER;"

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
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://admin:admin@localhost:5432/airflow_db
MLFLOW_TRACKING_URI=file://$PROJECT_DIR/mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=$PROJECT_DIR/mlflow_artifacts
EOF

# Source environment variables
source .env

# Update airflow.cfg to use absolute path and disable examples
echo "🔧 Updating Airflow configuration..."
sed -i "s|dags_folder = .*|dags_folder = $PROJECT_DIR/airflow/dags|" airflow.cfg
sed -i "s|load_examples = .*|load_examples = False|" airflow.cfg
sed -i "s|dags_are_paused_at_creation = .*|dags_are_paused_at_creation = False|" airflow.cfg

# Update database connection to PostgreSQL
echo "🔧 Configuring PostgreSQL connection..."
sed -i "s|sql_alchemy_conn = .*|sql_alchemy_conn = postgresql+psycopg2://admin:admin@localhost:5432/airflow_db|" airflow.cfg

# Ensure executor is set to LocalExecutor
sed -i "s|executor = .*|executor = LocalExecutor|" airflow.cfg

# Verify critical configurations
echo "📋 Verifying configuration..."
echo "DAGs folder: $(grep 'dags_folder' airflow.cfg)"
echo "Load examples: $(grep 'load_examples' airflow.cfg)"
echo "Database: $(grep 'sql_alchemy_conn' airflow.cfg)"
echo "Executor: $(grep '^executor' airflow.cfg)"

# Initialize Airflow database with PostgreSQL
echo "🔧 Initializing Airflow database..."
# Remove any existing SQLite database
rm -f airflow.db
airflow db init

# Create admin user
echo "👤 Creating Airflow admin user (admin/admin)..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || echo "Admin user already exists"

# Verify DAG is detected
echo "🔍 Verifying DAG detection..."
if airflow dags list | grep -q stellar; then
    echo "✅ Stellar DAG detected successfully!"
else
    echo "⚠️ Stellar DAG not detected - checking DAG file..."
    if [ -f "airflow/dags/stellar_pipeline_dag.py" ]; then
        echo "✅ DAG file exists at airflow/dags/stellar_pipeline_dag.py"
        echo "🔍 Testing DAG import..."
        python -c "
import sys, os
sys.path.append('$PROJECT_DIR/src')
os.chdir('$PROJECT_DIR')
try:
    from airflow.dags.stellar_pipeline_dag import dag
    print('✅ DAG imported successfully!')
except Exception as e:
    print(f'❌ DAG import failed: {e}')
"
    else
        echo "❌ DAG file not found! Please check airflow/dags/stellar_pipeline_dag.py"
    fi
fi

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
echo ""
echo "📊 Your stellar_classification_pipeline DAG will be visible in Airflow UI!"
echo "🔑 Database: PostgreSQL with admin/admin credentials"