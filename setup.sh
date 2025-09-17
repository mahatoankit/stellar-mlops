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

# Verify DAG file exists
if [ ! -f "airflow/dags/stellar_pipeline_dag.py" ]; then
    echo "❌ ERROR: DAG file not found at airflow/dags/stellar_pipeline_dag.py"
    echo "Please ensure the DAG file exists before running setup."
    exit 1
fi

# Set environment variables
export AIRFLOW_HOME=$PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

# Generate fresh, clean Airflow configuration
echo "🔧 Generating clean Airflow configuration..."
rm -f airflow.cfg airflow.db

# Create minimal, production-ready airflow.cfg
cat > airflow.cfg << EOF
[core]
dags_folder = $PROJECT_DIR/airflow/dags
executor = LocalExecutor
load_examples = False
dags_are_paused_at_creation = False
auth_manager = airflow.auth.managers.fab.fab_auth_manager.FabAuthManager

[database]
sql_alchemy_conn = postgresql+psycopg2://admin:admin@localhost:5432/airflow_db

[logging]
base_log_folder = $PROJECT_DIR/logs

[webserver]
web_server_port = 8080
expose_config = False

[scheduler]
dag_dir_list_interval = 300
EOF

echo "✅ Clean Airflow configuration created"

# Initialize Airflow database with PostgreSQL
echo "🔧 Initializing Airflow database..."
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

# Test database connection
echo "🔍 Testing database connection..."
python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        database='airflow_db',
        user='admin',
        password='admin'
    )
    print('✅ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

# Verify DAG is detected
echo "🔍 Verifying DAG detection..."
if airflow dags list | grep -q stellar; then
    echo "✅ Stellar DAG detected successfully!"
    airflow dags unpause stellar_classification_pipeline || echo "DAG already unpaused"
else
    echo "⚠️ Stellar DAG not detected - will be available after starting services"
fi

# Set permissions
chmod +x start.sh stop.sh

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the pipeline:"
echo "   ./start.sh"
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
echo ""
echo "⚠️  Important: This is a LOCAL Python pipeline (no Docker/Kubernetes required)"