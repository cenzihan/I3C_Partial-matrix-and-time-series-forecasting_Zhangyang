#!/bin/bash

# =================================================================================
# Script to start the CSI reconstruction and prediction model training.
#
# Usage:
#   ./start_training.sh
#
# Description:
#   This script sets the visible CUDA devices, activates the 'zhangyang'
#   conda environment, and then executes the main PyTorch training script 'train.py'.
#   All model parameters and training configurations are managed
#   through 'config.yaml'.
# =================================================================================

echo "Starting PyTorch model training using 'zhangyang' environment..."

# Set the path to the Python executable within the target Conda environment.
# This is a more robust method than trying to activate the environment in a script.
CONDA_PYTHON_EXEC="/data/user/cenzihan/.conda/envs/zhangyang/bin/python"

# Check if the Python executable exists
if [ ! -f "$CONDA_PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $CONDA_PYTHON_EXEC"
    echo "Please ensure the 'zhangyang' conda environment exists and is correctly located."
    exit 1
fi

# Set the visible GPU devices. Modify this line to change GPU allocation.
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Run the training script using the specific Python interpreter.
# The --config argument points to the YAML file containing our parameters.
"$CONDA_PYTHON_EXEC" src/train.py --config config.yaml

echo "Training script finished."
