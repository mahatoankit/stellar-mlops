#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - STOP SERVICES (DOCKER)
# =============================================================================

echo "🛑 Stopping Stellar Classification MLOps Pipeline..."

# Stop all Docker containers
docker-compose down

echo "✅ All services stopped!"
