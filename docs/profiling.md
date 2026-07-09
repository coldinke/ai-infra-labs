# Profiling Workflow

Use benchmarks to measure latency and profilers to explain it.

## Tool Roles

`nsys` is NVIDIA Nsight Systems. It answers:

- how many CUDA kernels were launched
- where time appears on the CPU/GPU timeline
- whether NVTX ranges cover the intended workload
- whether synchronization or launch overhead dominates

`ncu` is NVIDIA Nsight Compute. It answers:

- why one kernel is slow
- memory throughput and cache behavior
- occupancy, register pressure, and shared memory usage
- warp stalls and scheduler behavior

Use `nsys` first. Use `ncu` after the hot kernel is known.

## Shared Runner

Local benchmark:

```bash
bash scripts/run_lab.sh --lab 000-cuda-rmsnorm --type rmsnorm -- --mode benchmark --variant all
```

Local Nsight Systems profile:

```bash
bash scripts/run_lab.sh --lab 000-cuda-rmsnorm --type rmsnorm --nsys -- --mode profile --variant scalar --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10
```

The runner writes to:

```text
labs/<lab>/profiles/<timestamp>-<type>-<run-kind>-<host>/
```

Each run contains:

- `env.md`: OS, GPU, CUDA, PyTorch, Nsight paths
- `result.txt`: benchmark or profiler command output
- `summary.md`: command and output location
- `nsys_stats.txt`: focused Nsight Systems reports when `--nsys` is used

## Artifact Policy

Raw profiler databases are useful but noisy and large:

- `.nsys-rep`
- `.sqlite`
- `.ncu-rep`
- `.qdstrm`

They are ignored by git and excluded from normal remote pull. Use
`just pull-raw` only when local GUI inspection is required.

Curated conclusions should be written into lab notes instead of relying on raw
profile directories.

## Nsight Compute

Example direct `ncu` command:

```bash
ncu \
  --target-processes all \
  --kernel-name regex:rmsnorm_f32_kernel \
  --launch-skip 5 \
  --launch-count 1 \
  --set full \
  python3 labs/000-cuda-rmsnorm/benchmarks/rmsnorm.py \
    --mode profile \
    --variant scalar \
    --batch-size 1024 \
    --hidden-size 8192 \
    --warmup 5 \
    --iters 10
```

Keep `iters` small under `ncu`. It may replay the target kernel many times to
collect hardware counters.

If `ncu` fails with `ERR_NVGPUCTRPERM`, the process cannot access NVIDIA GPU
performance counters. This is common in managed cloud containers where root is
container root, not host driver administrator.
