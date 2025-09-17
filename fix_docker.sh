#!/bin/bash
# Fix Docker environment issues

echo "🔧 Fixing Docker environment..."

# Clear all Docker environment variables
unset DOCKER_HOST
unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH
unset DOCKER_API_VERSION
unset DOCKER_CONTEXT
unset COMPOSE_HOST
unset COMPOSE_TLS_VERSION

# Set proper Docker socket
export DOCKER_HOST=unix:///var/run/docker.sock

# Check if Docker daemon is running
if ! systemctl is-active --quiet docker; then
    echo "🔄 Starting Docker daemon..."
    sudo systemctl start docker
    sleep 3
fi

# Test Docker connection
echo "🧪 Testing Docker connection..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker is working correctly"
else
    echo "❌ Docker connection still has issues"
    echo "🔧 Trying to restart Docker service..."
    sudo systemctl restart docker
    sleep 5
fi

# Check containers
echo "📊 Checking container status..."
docker ps --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "✅ Docker environment fixed!"
echo "Now you can use: docker-compose ps"