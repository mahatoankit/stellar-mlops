#!/bin/bash
# =============================================================================
# STELLAR CLASSIFICATION MLOPS PIPELINE - DEBUG SERVICES
# =============================================================================

echo "🔍 STELLAR CLASSIFICATION PIPELINE - DEBUG INFORMATION"
echo "============================================================"
echo ""

# System Information
echo "📊 SYSTEM INFORMATION:"
echo "   OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
echo "   Kernel: $(uname -r)"
echo "   Architecture: $(uname -m)"
echo "   Available Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "   Available Disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# Docker Information
echo "🐳 DOCKER INFORMATION:"
echo "   Docker Version: $(docker --version 2>/dev/null || echo "Docker not found")"
echo "   Docker Compose Version: $(docker compose version 2>/dev/null || echo "Docker Compose not found")"
echo "   Docker Status: $(systemctl is-active docker 2>/dev/null || echo "Unknown")"
echo ""

# Port Usage
echo "🔌 PORT USAGE CHECK:"
echo "   Port 8000 (FastAPI): $(sudo netstat -tlnp 2>/dev/null | grep :8000 || echo "Not in use")"
echo "   Port 8080 (Airflow): $(sudo netstat -tlnp 2>/dev/null | grep :8080 || echo "Not in use")"
echo "   Port 5000 (MLflow): $(sudo netstat -tlnp 2>/dev/null | grep :5000 || echo "Not in use")"
echo "   Port 3306 (MariaDB): $(sudo netstat -tlnp 2>/dev/null | grep :3306 || echo "Not in use")"
echo ""

# Container Status
echo "📦 CONTAINER STATUS:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers found"
echo ""

# Service Health Checks
echo "🏥 SERVICE HEALTH CHECKS:"
echo "   FastAPI Health: $(curl -s -f http://localhost:8000/health > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Not responding")"
echo "   Airflow Health: $(curl -s -f http://localhost:8080/health > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Not responding")"
echo "   MLflow Health: $(curl -s -f http://localhost:5000/ > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Not responding")"
echo ""

# Container Logs (Last 20 lines each)
echo "📝 RECENT CONTAINER LOGS:"
echo ""

if docker ps -q --filter "name=stellar-fastapi" | grep -q .; then
    echo "🔍 FastAPI Logs (last 20 lines):"
    docker logs --tail 20 stellar-fastapi 2>&1 | sed 's/^/   /'
    echo ""
else
    echo "❌ FastAPI container not running"
    echo ""
fi

if docker ps -q --filter "name=airflow-standalone" | grep -q .; then
    echo "🔍 Airflow Logs (last 20 lines):"
    docker logs --tail 20 airflow-standalone 2>&1 | sed 's/^/   /'
    echo ""
else
    echo "❌ Airflow container not running"
    echo ""
fi

if docker ps -q --filter "name=stellar-mariadb" | grep -q .; then
    echo "🔍 MariaDB Logs (last 20 lines):"
    docker logs --tail 20 stellar-mariadb 2>&1 | sed 's/^/   /'
    echo ""
else
    echo "❌ MariaDB container not running"
    echo ""
fi

# Network Connectivity Test
echo "🌐 NETWORK CONNECTIVITY TEST:"
if docker ps -q --filter "name=stellar-fastapi" | grep -q .; then
    echo "   FastAPI -> MariaDB: $(docker exec stellar-fastapi ping -c 1 mariadb-columnstore > /dev/null 2>&1 && echo "✅ Connected" || echo "❌ Cannot reach")"
else
    echo "   FastAPI -> MariaDB: ❌ FastAPI container not running"
fi
echo ""

# File Permissions Check
echo "📁 FILE PERMISSIONS CHECK:"
echo "   Data directory: $(ls -ld data 2>/dev/null | awk '{print $1, $3, $4}' || echo "Not found")"
echo "   Models directory: $(ls -ld models 2>/dev/null | awk '{print $1, $3, $4}' || echo "Not found")"
echo "   API directory: $(ls -ld api 2>/dev/null | awk '{print $1, $3, $4}' || echo "Not found")"
echo ""

echo "============================================================"
echo "🚨 TROUBLESHOOTING TIPS:"
echo ""
echo "If FastAPI is not working:"
echo "1. Check the logs above for Python/pip installation errors"
echo "2. Ensure sufficient disk space and memory"
echo "3. Try rebuilding containers: docker compose down && docker compose up -d"
echo "4. Check if port 8000 is blocked by firewall"
echo ""
echo "If services fail to start:"
echo "1. Run: docker compose down --volumes"
echo "2. Run: docker system prune -f"
echo "3. Run: ./start.sh"
echo ""
echo "For permission issues:"
echo "1. Run: sudo chmod -R 777 data models"
echo "2. Run: docker compose restart"
echo "============================================================"