#!/bin/bash
# Activate the stellar-mlops environment and set variables
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stellar-mlops
source "$PROJECT_DIR/.env"
echo "✅ Environment activated. Project directory: $PROJECT_DIR"
