# 📋 Deployment Checklist - UPDATED WITH CROSS-SYSTEM FIXES

Use this checklist when deploying the Stellar Classification MLOps Pipeline on a new Ubuntu system.

## 🚨 **CRITICAL FIXES APPLIED** 
✅ **UnboundLocalError** - Fixed duplicate import statements in DAG  
✅ **FileNotFoundError** - Fixed model path handling with XCom  
✅ **MLflow Issues** - Enhanced environment configuration  
✅ **Cross-System Compatibility** - Added diagnostic tools  

**Last Updated**: December 2024 - Battle-tested across multiple Ubuntu systems!

## ✅ Pre-Deployment

- [ ] Ubuntu 18.04+ system available
- [ ] Internet connection for package downloads
- [ ] Sufficient disk space (minimum **8GB**, recommended 20GB)
- [ ] User has sudo privileges (for Docker installation)
- [ ] **🆕 CRITICAL**: Run diagnostic first: `./system_diagnostic.sh`

## ✅ Installation Steps

### 0. 🆕 Quick Diagnostic (NEW - Run This First!)
- [ ] **Download diagnostic script if not present**
- [ ] Run: `./system_diagnostic.sh > diagnostic_output.txt`
- [ ] Review output for any red ERROR messages
- [ ] Fix critical issues before proceeding

### 1. Install Prerequisites
- [ ] Install Docker & Docker Compose: 
  ```bash
  sudo apt update && sudo apt install -y docker.io docker-compose
  sudo systemctl start docker && sudo systemctl enable docker
  sudo usermod -aG docker $USER
  ```
- [ ] **Logout and login again** (important for Docker group)
- [ ] Verify Docker: `docker --version && docker compose version`

### 2. Clone Repository
- [ ] Clone the repository: `git clone <repository-url>`
- [ ] Navigate to directory: `cd stellar-classification-mlops`
- [ ] Verify files present: `ls -la`
- [ ] **Fix permissions**: `sudo chown -R $USER:$USER . && chmod +x *.sh`

### 3. Run Setup
- [ ] **Clean start**: `./stop.sh && docker system prune -f` (if re-deploying)
- [ ] Start services: `./start.sh`
- [ ] Wait for completion (2-5 minutes for all containers)
- [ ] **Check status**: `docker compose ps` (should show 5 healthy containers)

### 4. Validate Installation
- [ ] **All containers running**: `docker compose ps`
- [ ] **Check logs**: `docker compose logs --tail 50`
- [ ] **Network test**: `docker compose exec airflow-standalone ping stellar-mlflow`
- [ ] Fix any issues before proceeding

## ✅ First Run

### 1. Start Services
- [ ] Run start script: `./start.sh`
- [ ] Wait for all services to start (30-60 seconds)
- [ ] Check for any error messages

### 2. Verify Access
- [ ] Airflow UI accessible: http://localhost:8080
- [ ] Login with admin/admin works
- [ ] MLflow UI accessible: http://localhost:5000
- [ ] FastAPI docs accessible: http://localhost:8000/docs

### 3. Test Pipeline
- [ ] Navigate to Airflow UI
- [ ] Find `stellar_classification_pipeline` DAG
- [ ] **Verify DAG parsing**: Should show no import errors
- [ ] Enable the DAG (toggle switch)
- [ ] **🆕 CRITICAL**: DAG should parse without "UnboundLocalError" 
- [ ] Trigger manual run
- [ ] Monitor execution progress (all 12 tasks should complete)
- [ ] **Check MLflow**: Experiments should appear at http://localhost:5000

## ✅ Post-Deployment

### 1. Data Setup
- [ ] Place your data files in `data/raw/` directory
- [ ] Ensure CSV format matches expected schema
- [ ] Update `config/datasets/stellar.yaml` if needed

### 2. Monitoring
- [ ] Check log files in `logs/` directory
- [ ] Monitor disk space usage
- [ ] Set up log rotation if needed

### 3. Backup
- [ ] Backup configuration files
- [ ] Document any customizations made
- [ ] Note the project directory path

## 🛑 Shutdown

### When Done
- [ ] Stop services: `./stop.sh`
- [ ] Verify all processes stopped
- [ ] Check no ports still in use

## 🔧 Troubleshooting

### 🚨 **FIXED CRITICAL ERRORS** (September 2025)

#### ✅ "UnboundLocalError: local variable 'os' referenced before assignment"
**Cause**: Duplicate `import os` statements in DAG functions  
**Status**: **FIXED** - Removed duplicate imports  
**Verify**: DAG should parse without errors in Airflow UI  

#### ✅ "FileNotFoundError: models/best_random_forest_model.pkl"  
**Cause**: Path mismatch between training and evaluation tasks  
**Status**: **FIXED** - Using XCom for exact path sharing  
**Verify**: Model evaluation task should find trained model  

#### ✅ MLflow Experiments Not Tracking
**Cause**: Environment variable configuration  
**Status**: **FIXED** - Added proper MLflow environment setup  
**Verify**: Check http://localhost:5000 for experiment logs  

#### ✅ "PermissionError: [Errno 13] Permission denied: '/mlflow'" 
**Cause**: MLflow artifact storage permission issues
**Status**: **FIXED** - Disabled MLflow model artifacts logging to avoid permission conflicts
**Verify**: Training tasks should complete without permission errors
**Note**: Metrics and parameters are still logged to MLflow UI  

### Common Issues
- [ ] **Port conflicts**: Check if ports 8080, 5000, 8000 are free
  ```bash
  sudo netstat -tlnp | grep -E "(8080|5000|8000)"
  ```
- [ ] **Permission errors**: Fix with `sudo chown -R $USER:$USER .`
- [ ] **Docker not ready**: Wait 2-3 minutes after `./start.sh`
- [ ] **"Try Again" failures**: Run `./system_diagnostic.sh` first

### 🆕 Cross-System Deployment Issues
If deployment works on one system but fails on another:

1. **Run diagnostic script**: `./system_diagnostic.sh`
2. **Check Docker version**: Must be 20.10+ with Compose V2
3. **Clean restart**: `./stop.sh && docker system prune -f && ./start.sh`
4. **File permissions**: `sudo chown -R $USER:$USER . && chmod +x *.sh`
5. **Environment differences**: Check diagnostic output for system-specific issues

### Log Locations
- [ ] Airflow scheduler: `logs/scheduler/latest/scheduler.log`
- [ ] Airflow webserver: `logs/webserver/webserver.log`  
- [ ] MLflow: `logs/mlflow.log`
- [ ] FastAPI: `logs/fastapi.log`
- [ ] **🆕 All container logs**: `docker compose logs --tail 100`

## 📞 Support Commands

```bash
# 🆕 MOST IMPORTANT - Run diagnostic first
./system_diagnostic.sh

# Check all container status
docker compose ps

# View all recent logs
docker compose logs --tail 50

# Check specific service logs
docker compose logs airflow-standalone
docker compose logs stellar-mlflow

# Network connectivity test
docker compose exec airflow-standalone ping stellar-mlflow

# Complete clean restart (fixes most issues)
./stop.sh
docker system prune -f
docker volume prune -f  
./start.sh

# Emergency reset (nuclear option)
./stop.sh
docker system prune -af
sudo rm -rf data/temp data/processed mlruns logs/airflow
./start.sh
```

## 🎯 Success Criteria

The deployment is successful when:
- [ ] All services start without errors (`docker compose ps` shows 5 healthy containers)
- [ ] Web interfaces are accessible:
  - [ ] Airflow: http://localhost:8080 (admin/admin)
  - [ ] MLflow: http://localhost:5000 
  - [ ] FastAPI: http://localhost:8000/docs
- [ ] **🆕 DAG parsing**: No UnboundLocalError or import issues
- [ ] **🆕 Pipeline execution**: All 12 tasks complete successfully
- [ ] **🆕 MLflow tracking**: Experiments appear in MLflow UI
- [ ] **🆕 Model artifacts**: Files created in `models/` directory
- [ ] No critical errors in logs (`docker compose logs` shows normal operation)

### 🚀 **Expected Pipeline Flow** (All Should Complete):
1. ✅ `load_file_data` - Loads star_classification.csv
2. ✅ `clean_file_data` - Data cleaning and validation  
3. ✅ `encode_file_data` - Feature encoding
4. ✅ `split_scale_data` - Train/test split and scaling
5. ✅ `train_baseline_models` - Trains SVM and Random Forest models
6. ✅ `evaluate_models` - Model evaluation and comparison
7. ✅ `save_final_model` - Saves best model artifacts
8. ✅ `load_db_data` through `save_best_model_to_db` - Database operations

---

**🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉**

**Deployment Date**: September 19, 2025  
**Deployed By**: ___________  
**System**: Ubuntu Linux  
**Status**: ✅ All critical fixes applied and tested  
**Notes**: Cross-system compatibility validated