
echo "Starting PyTorch model training using 'zhangyang' environment..."

CONDA_PYTHON_EXEC="/data/user/cenzihan/.conda/envs/zhangyang/bin/python"

if [ ! -f "$CONDA_PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $CONDA_PYTHON_EXEC"
    echo "Please ensure the 'zhangyang' conda environment exists and is correctly located."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=4

"$CONDA_PYTHON_EXEC" src/train.py --config config.yaml

