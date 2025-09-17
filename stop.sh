#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - STOP SCRIPT
# =============================================================================

echo "🛑 Stopping Stellar Classification MLOps Pipeline..."

# Stop services using PIDs if available
if [ -f ".pids" ]; then
    echo "📋 Stopping services using stored PIDs..."
    while read -r pid; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "🔸 Stopping process $pid..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done < <(tr ' ' '\n' < .pids)
    rm -f .pids
    sleep 3
fi

# Force kill any remaining Airflow processes
echo "🧹 Cleaning up remaining processes..."
pkill -f "airflow" || true
pkill -f "mlflow" || true
pkill -f "uvicorn.*stellar_main" || true

# Wait a moment for graceful shutdown
sleep 2

# Force kill if still running
pkill -9 -f "airflow" 2>/dev/null || true
pkill -9 -f "mlflow" 2>/dev/null || true
pkill -9 -f "uvicorn.*stellar_main" 2>/dev/null || true

# Clean up temporary files
echo "🧽 Cleaning up temporary files..."
rm -f airflow-scheduler.* airflow-webserver.*
rm -f .pids

echo "✅ All services stopped successfully!"
echo ""
echo "🔄 To restart the pipeline:"
echo "   ./start.sh"
echo ""
echo "🔧 To set up from scratch:"
echo "   ./setup.sh"
