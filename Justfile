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

# Run one lab's pytest suite locally.
test lab *args:
    python3 -m pytest "labs/{{lab}}/tests" {{args}}

# Run one lab benchmark locally with the conventional benchmark mode.
bench lab exp_type *args:
    @just run "{{lab}}" "{{exp_type}}" --mode benchmark {{args}}

# Run a local Nsight Systems profile through the shared lab runner.
profile-nsys lab exp_type *args:
    bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" --nsys -- {{args}}

# Run one lab Nsight Systems profile locally with the conventional profile mode.
profile lab exp_type *args:
    @just profile-nsys "{{lab}}" "{{exp_type}}" --mode profile {{args}}

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

# Run one lab's pytest suite on the remote GPU machine.
remote-test lab *args:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just remote-test" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      python3 -m pytest "labs/{{lab}}/tests" {{args}}

# Run one lab benchmark on the remote GPU machine with the conventional benchmark mode.
remote-bench lab exp_type *args:
    @just remote-run "{{lab}}" "{{exp_type}}" --mode benchmark {{args}}

# Run a lab Nsight Systems profile on the remote GPU machine.
remote-profile-nsys lab exp_type *args:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just remote-profile-nsys" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash scripts/run_lab.sh --lab "{{lab}}" --type "{{exp_type}}" --nsys -- {{args}}

# Run one lab Nsight Systems profile on the remote GPU machine with the conventional profile mode.
remote-profile lab exp_type *args:
    @just remote-profile-nsys "{{lab}}" "{{exp_type}}" --mode profile {{args}}

# Build the current RMSNorm CUDA extension on the remote GPU machine.
rmsnorm-build:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just rmsnorm-build" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash -lc 'cd labs/000-cuda-rmsnorm && python3 setup.py build_ext --inplace'

# Run the current RMSNorm CUDA smoke test on the remote GPU machine.
rmsnorm-test *args:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just rmsnorm-test" >&2; exit 1)
    @just remote-test 000-cuda-rmsnorm {{args}}

# Run the current RMSNorm CUDA baseline benchmark on the remote GPU machine.
rmsnorm-bench:
    @just remote-bench 000-cuda-rmsnorm rmsnorm --variant all

# Run the current RMSNorm CUDA baseline Nsight Systems profile.
rmsnorm-profile *args:
    @just remote-profile 000-cuda-rmsnorm rmsnorm --variant scalar --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10 {{args}}

# Build the current Softmax CUDA extension on the remote GPU machine.
softmax-build:
    @test -n "${GPU_HOST:-}" || (echo "Set GPU_HOST=user@host before using just softmax-build" >&2; exit 1)
    bash scripts/gpu_remote_sync.sh exec "$GPU_HOST" --port "{{port}}" --remote-dir "{{remote_dir}}" -- \
      bash -lc 'cd labs/001-cuda-softmax && python3 setup.py build_ext --inplace'

# Run the current Softmax pytest suite on the remote GPU machine.
softmax-test *args:
    @just remote-test 001-cuda-softmax {{args}}

# Run the current Softmax benchmark on the remote GPU machine.
softmax-bench:
    @just remote-bench 001-cuda-softmax softmax --variant all

# Run the current Softmax Nsight Systems profile.
softmax-profile *args:
    @just remote-profile 001-cuda-softmax softmax --variant cuda_naive --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10 {{args}}
