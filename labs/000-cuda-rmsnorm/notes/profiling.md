# Lab 000 Profiling Notes

## Environment

Remote run:

- GPU: NVIDIA GeForce RTX 4090
- OS: Ubuntu 22.04 container
- PyTorch: 2.8.0+cu128
- PyTorch CUDA: 12.8
- `nvcc`: `/usr/local/cuda-12.8/bin/nvcc`
- `nsys`: `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`
- `ncu`: `/opt/nvidia/nsight-compute/2025.1.1/ncu`

## Benchmark Summary

`cuda_min_ms` on the remote RTX 4090:

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

The CUDA baseline is correct within approximately `3e-6` max absolute
difference for the tested fp32 shapes.

## Nsight Systems Summary

Profile shape:

```text
B=32, H=8192, warmup=5, iters=10
block_size=256
```

PyTorch eager:

- NVTX range: `849,720 ns` for 10 profiled iterations
- `cudaLaunchKernel`: 92 calls in the full process trace
- Kernel pattern: roughly 6 kernels per RMSNorm call
- Main kernels: `pow`, reduction/mean, add, `rsqrt`, and elementwise multiply

CUDA baseline:

- NVTX range: `140,917 ns` for 10 profiled iterations
- `cudaLaunchKernel`: 17 calls in the full process trace
- Custom RMSNorm kernel: 15 instances in the full process trace
- The 15 custom instances come from `warmup=5 + iters=10`

The profile-level NVTX speedup is about `6.0x`. The benchmark-level
`cuda_min_ms` speedup at `B=32,H=8192` is about `8.77x`. This mismatch is
expected because Nsight tracing adds overhead and includes CPU-side launch and
timeline costs, while CUDA events measure elapsed stream time more directly.

## Interpretation

The CUDA baseline wins because it fuses the eager PyTorch expression into one
kernel. PyTorch eager launches multiple kernels and materializes intermediate
tensors. The custom kernel performs reduction, normalization, weighting, and
output write in one launch.

Small and medium shapes are mostly launch-overhead bound. That explains why the
CUDA baseline stays near 5.7 to 5.9 us for many shapes. Larger batch sizes begin
to scale with memory traffic.

## Next Optimization Targets

- Add vectorized load/store paths such as `float4` where alignment and hidden
  size allow it.
- Replace the simple shared-memory reduction with warp-level reduction plus a
  smaller block-level reduction.
- Tune block size for `H=4096` and `H=8192`.
- Consider shape-specialized kernels for common LLM hidden sizes.
- Use Nsight Compute when GPU performance counters are available.

## Nsight Compute Status

An initial `ncu --set full` attempt failed with:

```text
ERR_NVGPUCTRPERM
```

This is a platform permission issue, not a kernel correctness issue. Continue
kernel iteration with benchmark and `nsys` until a machine with accessible GPU
performance counters is available.
