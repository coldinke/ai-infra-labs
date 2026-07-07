# Lab Run Summary

- Lab: `labs/000-cuda-rmsnorm`
- Type: `torch-baseline`
- Run kind: `nsys`
- Time: `2026-07-07T11:09:19+08:00`
- Host: `autodl-container-7e7145a2ed-ff8a114c`
- Git branch: `main`
- Git commit: `ddf5241`

## Files

- Environment: `env.md`
- Result: `result.txt`
- Nsight Systems stats: `nsys_stats.txt`

## Command

```bash
nsys profile --trace=cuda\,nvtx\,osrt --output=profile --force-overwrite=true python3 /root/ai-infra-labs/labs/000-cuda-rmsnorm/benchmarks/torch_baseline.py --mode profile --batch-size 32 --hidden-size 8192 --warmup 5 --iters 10 
```

## Notes

TBD.
