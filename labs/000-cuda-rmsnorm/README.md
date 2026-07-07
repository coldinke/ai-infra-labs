# Lab 000: CUDA RMSNorm

## Goal

Implement and profile RMSNorm for LLM-style hidden sizes.

## Background

RMSNorm is commonly used in modern LLMs. This lab starts from a PyTorch baseline, then implements a naive CUDA kernel and gradually optimizes it using profiler feedback.

## Tasks

- [x] Implement PyTorch baseline
- [x] Implement naive CUDA version
- [x] Add correctness tests
- [x] Add benchmark script
- [x] Profile with Nsight Systems
- [ ] Profile with Nsight Compute
- [x] Write first-pass profiling notes

## Shapes

| batch_size | hidden_size | dtype |
|---|---|---|
| 1 | 1024 | fp32 |
| 1 | 4096 | fp32 |
| 8 | 4096 | fp32 |
| 32 | 4096 | fp32 |
| 32 | 8192 | fp32 |
| 128 | 8192 | fp32 |
| 256 | 8192 | fp32 |
| 512 | 8192 | fp32 |
| 1024 | 8192 | fp32 |

## Run

Build and test on a GPU machine:

```bash
cd labs/000-cuda-rmsnorm
python3 setup.py build_ext --inplace
python3 tests/test_rmsnorm.py
```

Run through the shared lab runner:

```bash
bash scripts/run_lab.sh --lab 000-cuda-rmsnorm --type torch-baseline -- --mode benchmark
bash scripts/run_lab.sh --lab 000-cuda-rmsnorm --type cuda-baseline -- --mode benchmark
```

With a configured remote GPU machine:

```bash
just push
just rmsnorm-build
just rmsnorm-test
just rmsnorm-bench
just rmsnorm-profile
just pull
```

## Benchmark Results

Environment for the first remote run:

- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.8.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Driver CUDA reported by `nvidia-smi`: 13.0

`cuda_min_ms` comparison:

| batch | hidden | block | PyTorch eager ms | CUDA baseline ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 256 | 0.040053 | 0.005825 | 6.88x |
| 1 | 4096 | 256 | 0.041693 | 0.005681 | 7.34x |
| 8 | 4096 | 256 | 0.050819 | 0.005812 | 8.74x |
| 32 | 4096 | 256 | 0.050946 | 0.005908 | 8.62x |
| 32 | 8192 | 256 | 0.050405 | 0.005747 | 8.77x |
| 128 | 8192 | 256 | 0.043596 | 0.005825 | 7.48x |
| 256 | 8192 | 256 | 0.042916 | 0.008010 | 5.36x |
| 512 | 8192 | 256 | 0.042850 | 0.013791 | 3.11x |
| 1024 | 8192 | 256 | 0.123472 | 0.024349 | 5.07x |

## Profiler Notes

See [notes/profiling.md](notes/profiling.md).

## Lessons Learned

- PyTorch eager RMSNorm decomposes into multiple CUDA kernels for this expression.
- The naive CUDA baseline is already much faster because it fuses RMSNorm into one kernel.
- Small and medium shapes are dominated by launch overhead; larger batch sizes begin to show memory traffic scaling.
- Nsight Systems is enough to validate launch count and timeline behavior.
- Nsight Compute is the next tool for kernel-internal bottlenecks, but cloud machines may block hardware performance counters.
