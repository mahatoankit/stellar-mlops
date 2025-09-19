#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS - SYSTEM DIAGNOSTIC SCRIPT
# =============================================================================
# Run this script on any system to diagnose deployment issues

echo "🔍 STELLAR CLASSIFICATION MLOPS - SYSTEM DIAGNOSTIC"
echo "============================================================"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

# System Information
echo "📊 SYSTEM INFORMATION:"
echo "   OS: $(uname -a)"
echo "   Available Memory: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $2}' || echo 'N/A')"
echo "   Available Disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# Docker Information
echo "🐳 DOCKER INFORMATION:"
echo "   Docker Version: $(docker --version 2>/dev/null || echo 'Docker not found')"
echo "   Docker Compose Version: $(docker compose version 2>/dev/null || echo 'Docker Compose not found')"
echo "   Docker Status: $(systemctl is-active docker 2>/dev/null || echo 'Unknown')"
echo ""

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker and try again."
    echo "   Try: sudo systemctl start docker"
    exit 1
fi

# File Permissions Check
echo "📁 FILE PERMISSIONS:"
echo "   Project directory: $(ls -ld . | awk '{print $1, $3, $4}')"
echo "   Scripts:"
for script in start.sh stop.sh debug_services.sh; do
    if [ -f "$script" ]; then
        echo "     $script: $(ls -l $script | awk '{print $1, $3, $4}')"
    else
        echo "     $script: ❌ Missing"
    fi
done
echo ""

# Required Files Check
echo "📋 REQUIRED FILES CHECK:"
required_files=(
    "docker-compose.yml"
    "Dockerfile" 
    "airflow/dags/stellar_pipeline_dag.py"
    "src/stellar_ingestion.py"
    "config/datasets/stellar.yaml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
    fi
done
echo ""

# Docker Services Check
echo "🚀 DOCKER SERVICES STATUS:"
if docker compose ps &> /dev/null; then
    docker compose ps
else
    echo "   ❌ No services running"
fi
echo ""

# Port Usage Check
echo "🔌 PORT USAGE:"
ports=(8000 8080 5000 3306 5432)
for port in "${ports[@]}"; do
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        service_info=$(netstat -tlnp 2>/dev/null | grep ":$port " | head -1)
        echo "   Port $port: 🟢 IN USE - $service_info"
    else
        echo "   Port $port: 🔴 Available"
    fi
done
echo ""

# Container Health Check
echo "🏥 CONTAINER HEALTH:"
containers=("airflow-standalone" "stellar-fastapi" "stellar-mlflow" "stellar-mariadb" "stellar-postgres")
for container in "${containers[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "$container"; then
        status=$(docker ps --filter "name=$container" --format "{{.Status}}")
        echo "   ✅ $container: $status"
    else
        echo "   ❌ $container: Not running"
    fi
done
echo ""

# Service Connectivity Test
echo "🌐 SERVICE CONNECTIVITY:"
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    echo "   Testing from Airflow container..."
    
    # Test MLflow
    if docker exec airflow-standalone curl -s -f http://stellar-mlflow:5000/health &>/dev/null; then
        echo "   ✅ MLflow: Reachable"
    else
        echo "   ❌ MLflow: Not reachable"
    fi
    
    # Test MariaDB
    if docker exec airflow-standalone ping -c 1 mariadb-columnstore &>/dev/null; then
        echo "   ✅ MariaDB: Network reachable"
    else
        echo "   ❌ MariaDB: Network not reachable"
    fi
    
    # Test PostgreSQL
    if docker exec airflow-standalone pg_isready -h stellar-postgres -U admin &>/dev/null; then
        echo "   ✅ PostgreSQL: Ready"
    else
        echo "   ❌ PostgreSQL: Not ready"
    fi
else
    echo "   ⚠️  Airflow container not running - skipping connectivity tests"
fi
echo ""

# Python Environment Check
echo "🐍 PYTHON ENVIRONMENT (in Airflow container):"
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    echo "   Python Version: $(docker exec airflow-standalone python --version 2>/dev/null || echo 'Unknown')"
    echo "   Working Directory: $(docker exec airflow-standalone pwd 2>/dev/null || echo 'Unknown')"
    
    # Test imports
    echo "   Testing Python imports:"
    if docker exec airflow-standalone python -c "import mlflow; print('MLflow version:', mlflow.__version__)" 2>/dev/null; then
        echo "     ✅ MLflow import successful"
    else
        echo "     ❌ MLflow import failed"
    fi
    
    if docker exec airflow-standalone python -c "from src.stellar_ingestion import train_stellar_models" 2>/dev/null; then
        echo "     ✅ stellar_ingestion import successful"
    else
        echo "     ❌ stellar_ingestion import failed"
    fi
else
    echo "   ⚠️  Airflow container not running"
fi
echo ""

# DAG Status Check
echo "📊 AIRFLOW DAG STATUS:"
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    echo "   Checking DAG..."
    if docker exec airflow-standalone airflow dags list 2>/dev/null | grep -q "stellar_classification_pipeline"; then
        echo "   ✅ stellar_classification_pipeline DAG found"
        
        # Check DAG parsing errors
        dag_errors=$(docker exec airflow-standalone airflow dags show stellar_classification_pipeline 2>&1 | grep -i error || echo "")
        if [ -z "$dag_errors" ]; then
            echo "   ✅ DAG parsing successful"
        else
            echo "   ❌ DAG parsing errors:"
            echo "$dag_errors" | sed 's/^/      /'
        fi
    else
        echo "   ❌ stellar_classification_pipeline DAG not found"
    fi
else
    echo "   ⚠️  Airflow container not running"
fi
echo ""

# Recent Error Logs
echo "📝 RECENT ERROR LOGS:"
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    echo "   Last 10 Airflow errors:"
    recent_errors=$(docker logs airflow-standalone 2>&1 | grep -i "error\|exception\|failed\|unboundlocalerror" | tail -10)
    if [ -z "$recent_errors" ]; then
        echo "   ✅ No recent errors found"
    else
        echo "$recent_errors" | sed 's/^/      /'
    fi
else
    echo "   ⚠️  Airflow container not running"
fi
echo ""

# Environment Variables
echo "🔧 ENVIRONMENT VARIABLES:"
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    echo "   MLflow/Airflow related variables:"
    docker exec airflow-standalone env | grep -E "(MLFLOW|AIRFLOW|POSTGRES|MARIADB)" | sort | sed 's/^/      /'
else
    echo "   ⚠️  Airflow container not running"
fi
echo ""

echo "============================================================"
echo "🎯 TROUBLESHOOTING RECOMMENDATIONS:"
echo ""

# Generate specific recommendations based on findings
if ! docker compose ps &> /dev/null || [ "$(docker compose ps --services 2>/dev/null | wc -l)" -lt 5 ]; then
    echo "🔴 CRITICAL: Start all services first:"
    echo "   ./start.sh"
    echo ""
fi

# Check for UnboundLocalError specifically
if docker ps --filter "name=airflow-standalone" --format "{{.Names}}" | grep -q "airflow-standalone"; then
    if docker logs airflow-standalone 2>&1 | grep -qi "unboundlocalerror.*os.*referenced.*before.*assignment"; then
        echo "🔴 FOUND: UnboundLocalError with 'os' variable"
        echo "   This indicates duplicate 'import os' statements in the DAG"
        echo "   The DAG has been fixed - restart Airflow:"
        echo "   docker compose restart airflow-standalone"
        echo ""
    fi
fi

echo "📋 STANDARD TROUBLESHOOTING STEPS:"
echo "1. 🔄 Clean restart: ./stop.sh && docker system prune -f && ./start.sh"
echo "2. 🔍 Check Airflow UI: http://localhost:8080 (admin/admin)"
echo "3. 🔍 Check MLflow UI: http://localhost:5000"
echo "4. 🔍 Check FastAPI: http://localhost:8000/docs"
echo "5. 📊 Trigger DAG manually in Airflow UI"
echo "6. 📝 Monitor task logs for specific errors"
echo ""

echo "🆘 If problems persist:"
echo "1. Share this diagnostic output"
echo "2. Check system requirements (8GB RAM, 20GB disk)"
echo "3. Verify Docker and Docker Compose versions"
echo ""

echo "============================================================"