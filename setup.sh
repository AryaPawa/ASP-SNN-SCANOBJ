#!/bin/bash
# setup.sh — One-command environment setup for ASP-SNN
# Usage: bash setup.sh

set -e

echo "============================================"
echo "  ASP-SNN Environment Setup"
echo "============================================"

# Create conda environment
if conda info --envs | grep -q "asp-snn"; then
    echo "[Setup] Environment 'asp-snn' already exists — updating..."
    conda env update -f environment.yml --prune
else
    echo "[Setup] Creating conda environment 'asp-snn'..."
    conda env create -f environment.yml
fi

echo ""
echo "[Setup] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate asp-snn

echo "[Setup] Verifying PyTorch + CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "[Setup] Creating directories..."
mkdir -p checkpoints logs data

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "    conda activate asp-snn"
echo "    python datasets/download.py --all"
echo "============================================"
