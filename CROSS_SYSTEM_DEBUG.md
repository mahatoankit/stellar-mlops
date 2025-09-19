# Cross-System Deployment Troubleshooting Guide

## Common Issues When "It Works on My Machine" But Not Others

### 1. Docker Environment Issues
**Problem**: Different Docker versions, cached images, or environment state
**Solution**:
```bash
# Complete cleanup and rebuild
./stop.sh
docker system prune -f
docker volume prune -f
./start.sh
```

### 2. File Permissions Issues
**Problem**: Different file ownership/permissions between systems
**Solution**:
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x *.sh
```

### 3. Python Import Path Issues
**Problem**: Different Python environments or import resolution
**Check**:
```bash
docker exec airflow-standalone python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')
print()
print('Can import stellar modules?')
try:
    from src.stellar_training import train_stellar_models
    print('✅ stellar_training imported successfully')
except Exception as e:
    print(f'❌ stellar_training import failed: {e}')
"
```

### 4. MLflow Connection Issues
**Problem**: Network connectivity or service startup timing
**Check**:
```bash
# Test MLflow connectivity
docker exec airflow-standalone python -c "
import requests
try:
    response = requests.get('http://stellar-mlflow:5000/health')
    print(f'✅ MLflow reachable: {response.status_code}')
except Exception as e:
    print(f'❌ MLflow not reachable: {e}')
"
```

### 5. Database Connection Issues
**Problem**: PostgreSQL/MariaDB not ready when Airflow starts
**Check**:
```bash
# Test database connections
docker exec airflow-standalone python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='stellar-postgres',
        database='airflow',
        user='airflow',
        password='airflow'
    )
    print('✅ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'❌ PostgreSQL connection failed: {e}')
"
```

### 6. Volume Mount Issues
**Problem**: Different file paths or missing directories
**Check**:
```bash
# Verify all required directories exist
docker exec airflow-standalone ls -la /opt/airflow/
docker exec airflow-standalone ls -la /opt/airflow/dags/
docker exec airflow-standalone ls -la /opt/airflow/src/
```

## Quick Diagnostic Script
Run this on the problematic system:

```bash
#!/bin/bash
echo "=== STELLAR MLOPS DIAGNOSTIC ==="
echo "Docker version: $(docker --version)"
echo "Docker Compose version: $(docker compose version)"
echo "System: $(uname -a)"
echo "User: $(whoami)"
echo ""

echo "=== CHECKING SERVICES ==="
docker compose ps

echo ""
echo "=== CHECKING AIRFLOW LOGS ==="
docker compose logs airflow-standalone | tail -20

echo ""
echo "=== CHECKING DAG PARSING ==="
docker exec airflow-standalone airflow dags list | grep stellar || echo "❌ DAG not found"

echo ""
echo "=== CHECKING FILE PERMISSIONS ==="
ls -la *.sh
ls -la airflow/dags/

echo ""
echo "=== DIAGNOSTIC COMPLETE ==="
```

## Environment Variables to Check
Different systems might have different environment variables:

```bash
# Check environment variables
docker exec airflow-standalone env | grep -E "(AIRFLOW|MLFLOW|POSTGRES|MARIADB)" | sort
```

## System-Specific Fixes

### Ubuntu/Debian Systems
```bash
# Install required packages
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
```

### CentOS/RHEL Systems
```bash
# Install Docker
sudo yum install -y docker docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

### macOS Systems
```bash
# Use Docker Desktop or Homebrew
brew install docker docker-compose
```

## Final Verification Steps
1. **Clean Environment**: `./stop.sh && docker system prune -f`
2. **Fresh Start**: `./start.sh`
3. **Check All Services**: `docker compose ps`
4. **Test DAG**: Access http://localhost:8080 and trigger the DAG
5. **Monitor Logs**: `docker compose logs -f airflow-standalone`