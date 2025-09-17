#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - START SCRIPT
# =============================================================================

set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🌟 Starting Stellar Classification MLOps Pipeline..."
echo "📁 Project directory: $PROJECT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Environment not configured. Please run ./setup.sh first."
    exit 1
fi

# Load environment variables
source .env

# Check if conda environment exists
if ! conda env list | grep -q "stellar-mlops"; then
    echo "❌ Conda environment 'stellar-mlops' not found. Please run ./setup.sh first."
    exit 1
fi

# Activate conda environment
echo "🔄 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stellar-mlops

# Stop any existing processes
echo "🛑 Stopping existing processes..."
pkill -f "airflow" || true
pkill -f "mlflow" || true
pkill -f "uvicorn.*stellar_main" || true
sleep 2

# Create log directories
mkdir -p logs/{scheduler,webserver}

# Start Airflow scheduler in background
echo "📋 Starting Airflow scheduler..."
nohup airflow scheduler > logs/scheduler/scheduler.log 2>&1 &
SCHEDULER_PID=$!

# Start Airflow webserver in background  
echo "🌐 Starting Airflow webserver..."
nohup airflow webserver --port 8080 > logs/webserver/webserver.log 2>&1 &
WEBSERVER_PID=$!

# Start MLflow server in background
echo "🧪 Starting MLflow server..."
nohup mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlflow_artifacts > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!

# Start FastAPI server in background
echo "🔌 Starting FastAPI server..."
nohup uvicorn api.stellar_main:app --host 0.0.0.0 --port 8000 --reload > logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Store PIDs for easy cleanup
echo "$SCHEDULER_PID $WEBSERVER_PID $MLFLOW_PID $FASTAPI_PID" > .pids

echo ""
echo "✅ All services started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Service URLs:"
echo "   📊 Airflow UI:  http://localhost:8080 (admin/admin)"
echo "   🧪 MLflow UI:   http://localhost:5000"
echo "   🔌 FastAPI:     http://localhost:8000"
echo "   📖 API Docs:    http://localhost:8000/docs"
echo ""
echo "📁 Log Files:"
echo "   📋 Scheduler:   logs/scheduler/scheduler.log"
echo "   🌐 Webserver:   logs/webserver/webserver.log"
echo "   🧪 MLflow:      logs/mlflow.log"
echo "   🔌 FastAPI:     logs/fastapi.log"
echo ""
echo "🛑 To stop all services: ./stop.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"