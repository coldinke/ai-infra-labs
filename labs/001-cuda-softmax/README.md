# 001 CUDA Softmax

This lab implements and profiles row-wise softmax for LLM-style shapes. The
initial version establishes the shared testing and benchmarking flow with a
PyTorch reference variant; CUDA and Triton variants can be added behind the
same `softmax_lab.py` variant interface.

## Scope

- [x] PyTorch reference baseline
- [x] Manual eager stable softmax baseline
- [x] CUDA extension scaffold
- [x] Naive CUDA baseline kernel
- [x] Reusable benchmark entrypoint
- [x] Correctness test scaffold
- [ ] Optimized CUDA variants
- [ ] Triton variant
- [ ] Nsight Systems and Nsight Compute profile notes

## Shapes

| Batch | Hidden |
|---:|---:|
| 1 | 1024 |
| 1 | 4096 |
| 8 | 4096 |
| 32 | 4096 |
| 32 | 8192 |
| 128 | 8192 |
| 256 | 8192 |
| 512 | 8192 |
| 1024 | 8192 |

## Commands

```bash
cd labs/001-cuda-softmax
python3 setup.py build_ext --inplace
```

```bash
python3 -m pytest labs/001-cuda-softmax/tests
```

```bash
bash scripts/run_lab.sh --lab 001-cuda-softmax --type softmax -- --mode benchmark --variant all
```

```bash
bash scripts/run_lab.sh \
  --lab 001-cuda-softmax \
  --type softmax \
  --nsys \
  -- \
  --mode profile \
  --variant cuda_naive \
  --batch-size 32 \
  --hidden-size 8192 \
  --warmup 5 \
  --iters 10
```

Remote shortcuts:

```bash
just softmax-build
just softmax-test
just softmax-bench
just softmax-profile
```
