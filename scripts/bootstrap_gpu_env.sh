#!/usr/bin/env bash
set -euo pipefail

python3 --version

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# PyTorch CUDA 12.8 build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install pytest numpy pandas matplotlib

python scripts/check_gpu.py