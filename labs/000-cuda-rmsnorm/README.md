# Lab 000: CUDA RMSNorm

## Goal

Implement and profile RMSNorm for LLM-style hidden sizes.

## Background

RMSNorm is commonly used in modern LLMs. This lab starts from a PyTorch baseline, then implements a naive CUDA kernel and gradually optimizes it using profiler feedback.

## Tasks

- [ ] Implement PyTorch baseline
- [ ] Implement naive CUDA version
- [ ] Add correctness tests
- [ ] Add benchmark script
- [ ] Profile with Nsight Systems
- [ ] Profile with Nsight Compute
- [ ] Write optimization notes

## Shapes

| batch_size | hidden_size | dtype |
|---|---|---|
| 1 | 1024 | fp32 |
| 1 | 4096 | fp32 |
| 8 | 4096 | fp32 |
| 32 | 4096 | fp32 |
| 32 | 8192 | fp32 |

## Benchmark Results

TBD.

## Profiler Notes

TBD.

## Lessons Learned

TBD.