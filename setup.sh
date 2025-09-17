#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - DOCKER SETUP
# =============================================================================

set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🌟 Setting up Stellar Classification MLOps Pipeline with Docker..."
echo "📁 Project directory: $PROJECT_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing Docker..."
    
    # Install Docker on Ubuntu
    sudo apt update
    sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    echo "✅ Docker installed successfully!"
    echo "⚠️  Please log out and log back in for Docker group changes to take effect"
    echo "   Or run: newgrp docker"
fi

# Check if Docker Compose is installed and working
if ! command -v docker-compose &> /dev/null || ! docker-compose --version &> /dev/null; then
    echo "📦 Installing/Upgrading Docker Compose..."
    
    # Remove old versions
    sudo rm -f /usr/local/bin/docker-compose
    sudo apt remove -y docker-compose &> /dev/null || true
    
    # Install latest Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    echo "✅ Docker Compose installed successfully!"
fi

# Ensure we're using the correct Docker Compose
export PATH="/usr/local/bin:$PATH"
echo "🔍 Docker Compose version: $(docker-compose --version)"

# Fix Docker permissions
echo "🔧 Configuring Docker permissions..."
sudo usermod -aG docker $USER

# Remove any problematic environment variables
unset DOCKER_HOST 2>/dev/null || true
unset DOCKER_TLS_VERIFY 2>/dev/null || true
unset DOCKER_CERT_PATH 2>/dev/null || true

# Verify Docker is running
if ! docker info &> /dev/null; then
    echo "🔄 Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
fi

# Create necessary directories for Docker volumes
echo "📁 Creating directories for Docker volumes..."
mkdir -p data/{raw,processed,temp,plots}
mkdir -p models
mkdir -p mlflow_artifacts
mkdir -p logs/{airflow,mlflow,fastapi}
mkdir -p postgres_data

# Set proper permissions for Docker volumes
sudo chown -R $USER:$USER data models mlflow_artifacts logs postgres_data

# Create environment file for Docker Compose
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

# Verify Docker Compose file exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ ERROR: docker-compose.yml not found!"
    echo "Please ensure docker-compose.yml exists in the project root."
    exit 1
fi

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Start services with Docker Compose
echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Initialize Airflow (if needed)
echo "🔧 Initializing Airflow in container..."
docker-compose exec airflow-webserver airflow db init || echo "Database already initialized"

# Create admin user
echo "👤 Creating Airflow admin user..."
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || echo "Admin user already exists"

# Verify DAG is detected
echo "🔍 Verifying DAG detection..."
docker-compose exec airflow-webserver airflow dags list | grep stellar && echo "✅ Stellar DAG detected!" || echo "⚠️ DAG will be available shortly"

echo "✅ Docker setup complete!"
echo ""
echo "🌐 Access points:"
echo "   - Airflow UI: http://localhost:8080 (admin/admin)"
echo "   - MLflow UI: http://localhost:5000"
echo "   - FastAPI: http://localhost:8000/docs"
echo ""
echo "🔧 Useful Docker commands:"
echo "   - Stop services: docker-compose down"
echo "   - View logs: docker-compose logs -f [service-name]"
echo "   - Restart services: docker-compose restart"
echo "   - Rebuild images: docker-compose build --no-cache"
echo ""
echo "📊 Your stellar_classification_pipeline DAG will be visible in Airflow UI!"
echo "🐳 All services are running in Docker containers for maximum portability!"