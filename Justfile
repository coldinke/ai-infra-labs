set shell := ["bash", "-uc"]
set dotenv-load := true

remote_dir := env_var_or_default("GPU_REMOTE_DIR", "ai-infra-labs")
lab := env_var_or_default("GPU_LAB", "all")
port := env_var_or_default("GPU_PORT", "22")

default:
    @just --list

# Run a benchmark script locally through the shared lab runner.
run lab exp_type *args:
    bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" -- {{args}}

# Run a local Nsight Systems profile through the shared lab runner.
profile-nsys lab exp_type *args:
    bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" --nsys -- {{args}}

# Sync local code using GPU_HOST.
push:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host or run: just push-remote user@host" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh push "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}"

# Sync local code to a remote GPU machine.
push-remote host port=port:
    bash scripts/gpu_remote_sync.sh push "{{host}}" --port "{{port}}" --remote-dir "{{remote_dir}}"

# Pull profiles using GPU_HOST.
pull:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host or run: just pull-remote user@host" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh pull "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}"

# Pull profiles and raw Nsight artifacts using GPU_HOST.
pull-raw:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just pull-raw" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh pull "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}" --raw

# Pull one lab's profile outputs using GPU_HOST.
pull-lab lab:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just pull-lab" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh pull "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}"

# Pull one lab's profile outputs and raw Nsight artifacts using GPU_HOST.
pull-lab-raw lab:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just pull-lab-raw" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh pull "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}" --raw

# Pull remote profile outputs back into this checkout.
pull-remote host port=port:
    bash scripts/gpu_remote_sync.sh pull "{{host}}" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}"

# Pull one lab's profile outputs back into this checkout.
pull-remote-lab host lab port=port:
    bash scripts/gpu_remote_sync.sh pull "{{host}}" --port "{{port}}" --remote-dir "{{remote_dir}}" --lab "{{lab}}"

# Test SSH connectivity using GPU_HOST.
check-remote:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just check-remote" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh check "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}"

# Run a command in the remote project root using GPU_HOST.
remote *cmd:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just remote" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- {{cmd}}

# Run a lab benchmark on the remote GPU machine.
remote-run lab exp_type *args:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just remote-run" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" -- {{args}}

# Run a lab Nsight Systems profile on the remote GPU machine.
remote-profile-nsys lab exp_type *args:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just remote-profile-nsys" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" --nsys -- {{args}}

# Build the current RMSNorm CUDA extension on the remote GPU machine.
rmsnorm-build:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just rmsnorm-build" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash -lc 'cd labs/000-cuda-rmsnorm && python3 setup.py build_ext --inplace'

# Run the current RMSNorm CUDA smoke test on the remote GPU machine.
rmsnorm-test:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just rmsnorm-test" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      python3 labs/000-cuda-rmsnorm/tests/test_rmsnorm.py

# Run the current RMSNorm CUDA baseline benchmark on the remote GPU machine.
rmsnorm-bench:
    @just remote-run 000-cuda-rmsnorm cuda-baseline --mode benchmark

# Run the current RMSNorm CUDA baseline Nsight Systems profile.
rmsnorm-profile:
    @just remote-profile-nsys 000-cuda-rmsnorm cuda-baseline --mode profile --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10
