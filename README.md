# AI Infra Labs

A collection of labs for learning CUDA, LLM inference, PyTorch extensions, Triton, vLLM, and performance profiling.

## Goals

- Build CUDA fundamentals through measurable labs
- Learn profiler-first optimization
- Implement LLM-related operators
- Connect custom kernels to PyTorch
- Understand inference serving performance

## Labs

| ID | Lab | Status | Focus |
|---|---|---|---|
| 000 | CUDA RMSNorm | In Progress | CUDA baseline, benchmark, profiler |
| 001 | CUDA Softmax | Planned | Reduction, numerical stability |
| 002 | CUDA RoPE | Planned | LLM position embedding |
| 003 | PyTorch Extension | Planned | C++/CUDA operator integration |
| 004 | vLLM Profiling | Planned | Serving benchmark and analysis |

## Principles

1. Correctness first
2. Benchmark before optimization
3. Profile before guessing
4. Record every result
5. Prefer small complete labs over large unfinished projects