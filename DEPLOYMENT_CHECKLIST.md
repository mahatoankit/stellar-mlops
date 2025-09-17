# 📋 Deployment Checklist

Use this checklist when deploying the Stellar Classification MLOps Pipeline on a new Ubuntu system.

## ✅ Pre-Deployment

- [ ] Ubuntu 18.04+ system available
- [ ] Internet connection for package downloads
- [ ] Sufficient disk space (minimum 2GB)
- [ ] User has sudo privileges (for conda installation if needed)

## ✅ Installation Steps

### 1. Install Prerequisites
- [ ] Install Git: `sudo apt update && sudo apt install git -y`
- [ ] Install Conda/Miniconda if not present
- [ ] Verify conda works: `conda --version`

### 2. Clone Repository
- [ ] Clone the repository: `git clone <repository-url>`
- [ ] Navigate to directory: `cd stellar-classification-mlops`
- [ ] Verify files present: `ls -la`

### 3. Run Setup
- [ ] Make setup script executable: `chmod +x setup.sh`
- [ ] Run setup: `./setup.sh`
- [ ] Wait for completion (may take 5-10 minutes)
- [ ] Verify no errors in output

### 4. Validate Installation
- [ ] Run validation: `./validate.sh`
- [ ] All checks should pass with ✅
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
- [ ] Enable the DAG (toggle switch)
- [ ] Trigger manual run
- [ ] Monitor execution progress

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

### Common Issues
- [ ] **Port conflicts**: Check if ports 8080, 5000, 8000 are free
- [ ] **Permission errors**: Ensure scripts are executable
- [ ] **Conda not found**: Install Miniconda first
- [ ] **Package errors**: Check internet connection and retry

### Log Locations
- [ ] Airflow scheduler: `logs/scheduler/scheduler.log`
- [ ] Airflow webserver: `logs/webserver/webserver.log`
- [ ] MLflow: `logs/mlflow.log`
- [ ] FastAPI: `logs/fastapi.log`

## 📞 Support Commands

```bash
# Check running processes
ps aux | grep -E "(airflow|mlflow|uvicorn)"

# Check port usage
netstat -tlnp | grep -E "(8080|5000|8000)"

# View recent logs
tail -f logs/scheduler/scheduler.log

# Restart services
./stop.sh && sleep 5 && ./start.sh

# Validate setup
./validate.sh
```

## 🎯 Success Criteria

The deployment is successful when:
- [ ] All services start without errors
- [ ] Web interfaces are accessible
- [ ] DAG appears in Airflow UI
- [ ] Pipeline can be triggered manually
- [ ] No critical errors in logs

---

**Deployment Date**: ___________  
**Deployed By**: ___________  
**System**: ___________  
**Notes**: ___________