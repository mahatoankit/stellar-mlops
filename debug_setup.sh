#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS - DEBUG & TROUBLESHOOTING SCRIPT
# =============================================================================

echo "🔍 Debugging Stellar Classification MLOps Setup..."
echo "=================================================="

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 Project Directory: $PROJECT_DIR"
echo ""

# Check environment
echo "🐍 Environment Status:"
echo "----------------------"
if command -v conda &> /dev/null; then
    echo "✅ Conda available"
    conda info --envs | grep stellar-mlops || echo "❌ stellar-mlops environment not found"
else
    echo "❌ Conda not available"
fi
echo ""

# Check PostgreSQL
echo "🐘 PostgreSQL Status:"
echo "--------------------"
if systemctl is-active --quiet postgresql; then
    echo "✅ PostgreSQL service running"
    sudo -u postgres psql -c "\l" | grep airflow_db && echo "✅ airflow_db exists" || echo "❌ airflow_db missing"
else
    echo "❌ PostgreSQL service not running"
fi
echo ""

# Check Airflow configuration
echo "⚙️ Airflow Configuration:"
echo "-------------------------"
if [ -f "airflow.cfg" ]; then
    echo "✅ airflow.cfg exists"
    echo "DAGs folder: $(grep 'dags_folder' airflow.cfg)"
    echo "Load examples: $(grep 'load_examples' airflow.cfg)"
    echo "Database: $(grep 'sql_alchemy_conn' airflow.cfg)"
    echo "Executor: $(grep '^executor' airflow.cfg)"
else
    echo "❌ airflow.cfg not found"
fi
echo ""

# Check DAG file
echo "📄 DAG File Status:"
echo "------------------"
if [ -f "airflow/dags/stellar_pipeline_dag.py" ]; then
    echo "✅ stellar_pipeline_dag.py exists"
    echo "File size: $(stat -c%s airflow/dags/stellar_pipeline_dag.py) bytes"
else
    echo "❌ stellar_pipeline_dag.py not found"
fi
echo ""

# Check if environment can be activated
echo "🔄 Environment Activation Test:"
echo "------------------------------"
if source $(conda info --base)/etc/profile.d/conda.sh && conda activate stellar-mlops; then
    echo "✅ Environment activation successful"
    
    # Set environment variables
    export AIRFLOW_HOME=$PROJECT_DIR
    export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
    
    echo "🔍 Testing DAG import..."
    python -c "
import sys, os
sys.path.append('$PROJECT_DIR/src')
os.chdir('$PROJECT_DIR')
try:
    from airflow.dags.stellar_pipeline_dag import dag
    print('✅ DAG imported successfully!')
    print(f'DAG ID: {dag.dag_id}')
    print(f'DAG description: {dag.description}')
except Exception as e:
    print(f'❌ DAG import failed: {e}')
    import traceback
    traceback.print_exc()
"
    
    echo "🔍 Testing Airflow DAG detection..."
    airflow dags list | grep stellar && echo "✅ Stellar DAG detected by Airflow" || echo "❌ Stellar DAG not detected by Airflow"
    
else
    echo "❌ Environment activation failed"
fi
echo ""

# Check required packages
echo "📦 Required Packages:"
echo "--------------------"
if conda activate stellar-mlops; then
    python -c "
packages = ['airflow', 'pandas', 'scikit-learn', 'mlflow', 'fastapi', 'psycopg2']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - missing')
"
fi
echo ""

# Check database connectivity
echo "🔗 Database Connectivity:"
echo "------------------------"
if conda activate stellar-mlops; then
    export AIRFLOW_HOME=$PROJECT_DIR
    echo "Testing Airflow database connection..."
    airflow db check && echo "✅ Database connection successful" || echo "❌ Database connection failed"
fi
echo ""

echo "🏁 Debug complete!"
echo ""
echo "💡 Common solutions:"
echo "   1. If PostgreSQL is not running: sudo systemctl start postgresql"
echo "   2. If database doesn't exist: Re-run ./setup.sh"
echo "   3. If DAG not detected: Check airflow/dags/stellar_pipeline_dag.py exists"
echo "   4. If import fails: Check src/ directory and PYTHONPATH"
echo "   5. If still using SQLite: Check airflow.cfg sql_alchemy_conn setting"