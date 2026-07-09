from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def benchmark_wall_time(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        fn(*args)

    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def benchmark_cuda_event(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(iters):
        fn(*args)

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters


def benchmark_cuda(
    fn: Callable[..., Any],
    *args: Any,
    rounds: int,
    warmup: int,
    iters: int,
    include_wall: bool = True,
) -> dict[str, float]:
    cuda_times = []
    wall_times = []

    for _ in range(rounds):
        if include_wall:
            wall_times.append(benchmark_wall_time(fn, *args, warmup=warmup, iters=iters))
        cuda_times.append(benchmark_cuda_event(fn, *args, warmup=warmup, iters=iters))

    result = {
        "cuda_min_ms": min(cuda_times),
        "cuda_median_ms": statistics.median(cuda_times),
    }

    if include_wall:
        result.update(
            {
                "wall_min_ms": min(wall_times),
                "wall_median_ms": statistics.median(wall_times),
            }
        )

    return result


def format_gbps(num_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    seconds = ms / 1000.0
    return num_bytes / seconds / 1e9

