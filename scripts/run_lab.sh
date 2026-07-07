#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_lab.sh --lab <lab_dir> --type <experiment_type>

Examples:
  bash scripts/run_lab.sh --lab labs/000-cuda-rmsnorm --type torch-baseline
  bash scripts/run_lab.sh --lab labs/000-cuda-rmsnorm --type cuda-baseline
  bash scripts/run_lab.sh --lab labs/000-cuda-rmsnorm --type optimized
USAGE
}

LAB_DIR=""
EXP_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lab)
      LAB_DIR="$2"
      shift 2
      ;;
    --type)
      EXP_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LAB_DIR" || -z "$EXP_TYPE" ]]; then
  usage
  exit 1
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Allow both:
#   --lab labs/000-cuda-rmsnorm
#   --lab 000-cuda-rmsnorm
if [[ "$LAB_DIR" != labs/* ]]; then
  LAB_DIR="labs/$LAB_DIR"
fi

LAB_PATH="$ROOT/$LAB_DIR"

if [[ ! -d "$LAB_PATH" ]]; then
  echo "lab directory not found: $LAB_PATH"
  exit 1
fi

BENCH_FILE="$LAB_PATH/benchmarks/${EXP_TYPE//-/_}.py"

if [[ ! -f "$BENCH_FILE" ]]; then
  echo "benchmark file not found: $BENCH_FILE"
  echo
  echo "Expected mapping:"
  echo "  --type torch-baseline -> benchmarks/torch_baseline.py"
  echo "  --type cuda-baseline  -> benchmarks/cuda_baseline.py"
  echo "  --type optimized      -> benchmarks/optimized.py"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
HOST="$(hostname)"
OUT_DIR="$LAB_PATH/profiles/${TS}-${EXP_TYPE}-${HOST}"

mkdir -p "$OUT_DIR"

ENV_FILE="$OUT_DIR/env.md"
RESULT_FILE="$OUT_DIR/result.txt"
SUMMARY_FILE="$OUT_DIR/summary.md"

echo "== Run Lab =="
echo "root:       $ROOT"
echo "lab:        $LAB_DIR"
echo "type:       $EXP_TYPE"
echo "bench file: $BENCH_FILE"
echo "out dir:    $OUT_DIR"
echo

{
  echo "# Environment"
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
  nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
  echo '```'
  echo

  echo "## NVIDIA GPU Query"
  echo '```text'
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit --format=csv 2>/dev/null || true
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
} | tee "$ENV_FILE"

echo
echo "== Benchmark =="
python3 "$BENCH_FILE" | tee "$RESULT_FILE"

cat > "$SUMMARY_FILE" <<SUMMARY
# Lab Run Summary

- Lab: \`$LAB_DIR\`
- Type: \`$EXP_TYPE\`
- Time: \`$(date -Is)\`
- Host: \`$HOST\`
- Git commit: \`$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\`

## Files

- Environment: \`env.md\`
- Result: \`result.txt\`

## Notes

TBD.
SUMMARY

echo
echo "== Done =="
echo "saved to: $OUT_DIR"
