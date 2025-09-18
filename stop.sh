#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - STOP SERVICES (DOCKER)
# =============================================================================

set -e

echo "🛑 Stopping Stellar Classification MLOps Pipeline..."

# Clean up Docker environment variables to prevent corruption
unset DOCKER_HOST 2>/dev/null || true
unset DOCKER_TLS_VERIFY 2>/dev/null || true
unset DOCKER_CERT_PATH 2>/dev/null || true
unset DOCKER_API_VERSION 2>/dev/null || true
unset DOCKER_CONTEXT 2>/dev/null || true

# Set clean Docker environment
export DOCKER_HOST=unix:///var/run/docker.sock

# Stop all Docker containers
docker compose down

echo "✅ All services stopped!"
