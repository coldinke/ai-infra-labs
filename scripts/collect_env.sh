#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT_DIR="$ROOT/docs/env"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname)"
OUT="$OUT_DIR/${TS}-${HOST}-gpu-env.md"

{
  echo "# GPU Environment"
  echo
  echo "- Captured at: $(date -Is)"
  echo "- Hostname: $(hostname)"
  echo "- Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "- Git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
  echo

  echo "## OS"
  echo '```text'
  lsb_release -a 2>/dev/null || cat /etc/os-release
  echo '```'
  echo

  echo "## CPU"
  echo '```text'
  lscpu | sed -n '1,25p'
  echo '```'
  echo

  echo "## Memory"
  echo '```text'
  free -h
  echo '```'
  echo

  echo "## Disk"
  echo '```text'
  df -h
  echo '```'
  echo

  echo "## NVIDIA SMI"
  echo '```text'
  nvidia-smi
  echo '```'
  echo

  echo "## NVIDIA GPU Query"
  echo '```text'
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit --format=csv || true
  echo '```'
  echo

  echo "## NVCC"
  echo '```text'
  nvcc --version 2>/dev/null || echo "nvcc not found"
  echo '```'
  echo

  echo "## Python / PyTorch"
  echo '```text'
  python3 - <<'PY'
import sys
print("python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"device {i}: {props.name}")
            print(f"  capability: {props.major}.{props.minor}")
            print(f"  total memory: {props.total_memory / 1024**3:.2f} GB")
except Exception as e:
    print("failed to import torch:", repr(e))
PY
  echo '```'
} | tee "$OUT"

echo
echo "saved: $OUT"
