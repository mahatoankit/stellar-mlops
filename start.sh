#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - START SERVICES
# =============================================================================

set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting Stellar Classification MLOps Pipeline..."
echo "📁 Project directory: $PROJECT_DIR"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "🔄 Starting Docker service..."
    sudo systemctl start docker
    sleep 3
fi

# Remove any problematic environment variables
unset DOCKER_HOST 2>/dev/null || true
unset DOCKER_TLS_VERIFY 2>/dev/null || true
unset DOCKER_CERT_PATH 2>/dev/null || true
unset DOCKER_API_VERSION 2>/dev/null || true
unset DOCKER_CONTEXT 2>/dev/null || true

# Force Docker to use default socket
export DOCKER_HOST=unix:///var/run/docker.sock

# Ensure we're using the correct Docker Compose
export PATH="/usr/local/bin:$PATH"

# Verify Docker Compose file exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ ERROR: docker-compose.yml not found!"
    echo "Please ensure docker-compose.yml exists in the project root."
    exit 1
fi

# Create directories if they don't exist
echo "📁 Ensuring directories exist..."
mkdir -p data/{raw,processed,temp,plots}
mkdir -p models mlflow_artifacts logs/{airflow,mlflow,fastapi} postgres_data

# Fix permissions for Airflow user (UID 50000)
echo "🔧 Setting proper permissions for Airflow user..."
sudo chown -R 50000:0 data models mlflow_artifacts logs postgres_data
sudo chmod -R 755 data models mlflow_artifacts logs postgres_data

# Ensure specific directories have write permissions
sudo chmod -R 777 data/temp data/processed data/plots logs

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating environment configuration..."
    cat > .env << EOF
# Stellar Classification MLOps Pipeline Environment
PROJECT_DIR=$PROJECT_DIR
POSTGRES_DB=airflow_db
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
MLFLOW_BACKEND_STORE_URI=postgresql://admin:admin@postgres:5432/mlflow_db
MLFLOW_ARTIFACT_ROOT=/opt/mlflow/artifacts
EOF
fi

# Check if containers are already running
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "⚠️  Some services are already running. Stopping them first..."
    docker-compose down 2>/dev/null || true
    sleep 5
fi

# Clean up any orphaned containers
echo "🧹 Cleaning up any orphaned containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Start services with Docker Compose
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 20

# Fix permissions after containers start (in case volumes get reset)
echo "🔧 Final permission fix after container startup..."
sudo chown -R 50000:0 data models mlflow_artifacts logs postgres_data
sudo chmod -R 777 data/temp data/processed data/plots

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

# Verify services are healthy
echo "🏥 Performing health checks..."

# Check PostgreSQL
echo -n "📊 PostgreSQL: "
if docker-compose exec -T postgres pg_isready -U admin &>/dev/null; then
    echo "✅ Ready"
else
    echo "❌ Not ready (may need more time)"
fi

# Check MariaDB and set up permissions
echo -n "🗄️  MariaDB: "
if docker-compose exec -T mariadb-columnstore mysql -u root -pstellar_password -e "SELECT 1;" &>/dev/null; then
    echo "✅ Ready - Setting up database permissions..."
    
    # Create database and user with proper permissions
    docker-compose exec -T mariadb-columnstore mysql -u root -pstellar_password << 'EOF'
CREATE DATABASE IF NOT EXISTS stellar_db;
CREATE USER IF NOT EXISTS 'stellar_user'@'%' IDENTIFIED BY 'stellar_user_password';
CREATE USER IF NOT EXISTS 'stellar_user'@'localhost' IDENTIFIED BY 'stellar_user_password';
GRANT ALL PRIVILEGES ON stellar_db.* TO 'stellar_user'@'%';
GRANT ALL PRIVILEGES ON stellar_db.* TO 'stellar_user'@'localhost';
FLUSH PRIVILEGES;

USE stellar_db;
CREATE TABLE IF NOT EXISTS stellar_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    obj_ID BIGINT NOT NULL,
    alpha DOUBLE,
    delta_coord DOUBLE,
    u DOUBLE,
    g DOUBLE,
    r DOUBLE,
    i DOUBLE,
    z DOUBLE,
    run_ID INT,
    rerun_ID INT,
    cam_col INT,
    field_ID INT,
    spec_obj_ID BIGINT,
    class VARCHAR(10),
    redshift DOUBLE,
    plate INT,
    MJD INT,
    fiber_ID INT,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    u_g_color DOUBLE,
    g_r_color DOUBLE,
    r_i_color DOUBLE,
    i_z_color DOUBLE,
    is_scored BOOLEAN DEFAULT FALSE,
    data_split VARCHAR(10)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(255),
    run_type VARCHAR(50),
    status VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF
    
    # Test user connection
    if docker-compose exec -T mariadb-columnstore mysql -u stellar_user -pstellar_user_password -e "SELECT 1;" &>/dev/null; then
        echo "✅ Database permissions configured successfully"
    else
        echo "⚠️  Database user permissions may need more time to propagate"

else
    echo "❌ Not ready (may need more time)"
fi

# Check MLflow
echo -n "📈 MLflow: "
if curl -s http://localhost:5000 &>/dev/null; then
    echo "✅ Ready"
else
    echo "⏳ Starting up..."
fi

# Check FastAPI
echo -n "🔌 FastAPI: "
if curl -s http://localhost:8000/docs &>/dev/null; then
    echo "✅ Ready"
else
    echo "⏳ Starting up..."
fi

# Check Airflow
echo -n "🌪️  Airflow: "
if curl -s http://localhost:8080 &>/dev/null; then
    echo "✅ Ready"
else
    echo "⏳ Starting up..."
fi

# Test file creation permissions
echo "🧪 Testing file permissions..."
if sudo -u "#50000" touch data/temp/test_permission.txt 2>/dev/null; then
    echo "✅ File permissions working correctly"
    sudo rm -f data/temp/test_permission.txt
else
    echo "⚠️  Permission issue detected - fixing..."
    sudo chmod -R 777 data
fi

echo ""
echo "✅ Services startup complete!"
echo ""
echo "🌐 Access Points:"
echo "   🌪️  Airflow UI: http://localhost:8080"
echo "       Username: admin"
echo "       Password: admin"
echo ""
echo "   📈 MLflow UI: http://localhost:5000"
echo "   🔌 FastAPI: http://localhost:8000/docs"
echo ""
echo "🔧 Useful Commands:"
echo "   📊 Check status: docker-compose ps"
echo "   📝 View logs: docker-compose logs -f [service-name]"
echo "   🛑 Stop services: docker-compose down"
echo "   🔄 Restart: docker-compose restart [service-name]"
echo ""
echo "🎯 Your stellar_classification_pipeline DAG should be visible in Airflow!"
echo "   Wait 1-2 minutes for all services to fully initialize."
echo ""
echo "🚨 If tasks still fail with permission errors, run:"
echo "   sudo chmod -R 777 data && docker-compose restart airflow-standalone"