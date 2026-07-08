# Environment Troubleshooting

## CUDA Compiler Is Not Found

Symptoms:

```text
nvcc not found
CUDA_HOME environment variable is not set
```

Preferred fixes:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

or:

```bash
export NVCC_BIN=/usr/local/cuda-12.8/bin/nvcc
```

`labs/000-cuda-rmsnorm/setup.py` also tries to infer `CUDA_HOME` from `NVCC_BIN`,
`PATH`, `/usr/local/cuda*`, `/opt/cuda*`, and `/opt/nvidia`.

## Nsight Systems Is Not On PATH

Symptoms:

```text
nsys: command not found
```

Find it:

```bash
find /opt/nvidia /usr/local/cuda* -type f -name nsys 2>/dev/null
```

Then either rely on `scripts/run_lab.sh`, which searches those locations, or set:

```bash
export NSYS_BIN=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
```

## Nsight Compute Counter Permission

Symptoms:

```text
ERR_NVGPUCTRPERM
```

This means `ncu` cannot read GPU hardware performance counters. On managed cloud
GPU containers this usually requires platform support or a different instance
type. If you control the host, the NVIDIA driver setting is
`NVreg_RestrictProfilingToAdminUsers=0`.

Do not block kernel development on this. Continue with benchmarks and `nsys`;
use `ncu` later on a machine that exposes counters.

## SSH ControlPath Too Long

Symptoms on macOS:

```text
ControlPath too long
```

`scripts/gpu_remote_sync.sh` uses `/tmp/gpu-sync.XXXXXX` for SSH connection
sharing to keep the socket path short.

## Remote Pull Looks Stuck

Normal `just pull` excludes raw profiler databases. If pull appears slow, check
whether raw files are being copied through `just pull-raw`.

Use:

```bash
just pull
```

for normal text summaries, and:

```bash
just pull-raw
```

only when `.nsys-rep`, `.sqlite`, or `.ncu-rep` is needed locally.

## Remote Git Metadata Is Unknown

The remote sync intentionally excludes `.git`. This avoids needing remote git
credentials. As a result, remote `env.md` may show unknown git branch or commit.
The source of truth remains the local checkout that performed `just push`.
