from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import _rmsnorm_cuda  # noqa: E402


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype for cuda baseline: {name}")
    return DTYPE_MAP[name]


def rmsnorm_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_cuda_block_size() -> int:
    return int(_rmsnorm_cuda.block_size())


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def benchmark_wall_time(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        fn(*args)

    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def benchmark_cuda_event(fn, *args, warmup: int, iters: int) -> float:
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


def benchmark(fn, *args, rounds: int, warmup: int, iters: int) -> dict[str, float]:
    wall_times = []
    cuda_event_times = []

    for _ in range(rounds):
        wall_times.append(benchmark_wall_time(fn, *args, warmup=warmup, iters=iters))
        cuda_event_times.append(benchmark_cuda_event(fn, *args, warmup=warmup, iters=iters))

    return {
        "wall_min_ms": min(wall_times),
        "wall_median_ms": statistics.median(wall_times),
        "cuda_min_ms": min(cuda_event_times),
        "cuda_median_ms": statistics.median(cuda_event_times),
    }


def estimate_rmsnorm_bytes(batch_size: int, hidden_size: int, dtype: torch.dtype) -> int:
    # Rough lower-bound traffic for fused RMSNorm:
    # x read once for reduction + x read again for output + weight read + y write.
    element_size = torch.empty((), dtype=dtype).element_size()
    return 4 * batch_size * hidden_size * element_size


def format_gbps(num_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    seconds = ms / 1000.0
    return num_bytes / seconds / 1e9


def print_environment() -> None:
    print("== Environment ==")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()


def print_kernel_config() -> None:
    print("== CUDA Kernel Config ==")
    print(f"block_size: {rmsnorm_cuda_block_size()}")
    print()


def run_benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = parse_dtype(args.dtype)

    print_environment()
    print_kernel_config()

    shapes = [
        (1, 1024),
        (1, 4096),
        (8, 4096),
        (32, 4096),
        (32, 8192),
        (128, 8192),
        (256, 8192),
        (512, 8192),
        (1024, 8192),
    ]

    print("== RMSNorm CUDA Baseline ==")
    print(
        f"{'batch':>8} "
        f"{'hidden':>8} "
        f"{'block':>8} "
        f"{'elements':>12} "
        f"{'dtype':>14} "
        f"{'wall_min_ms':>14} "
        f"{'wall_med_ms':>14} "
        f"{'cuda_min_ms':>14} "
        f"{'cuda_med_ms':>14} "
        f"{'approx_GB/s':>14} "
        f"{'max_abs_diff':>14}"
    )

    for batch_size, hidden_size in shapes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)

        expected = rmsnorm_torch(x, weight, args.eps)
        actual = rmsnorm_cuda(x, weight, args.eps)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        max_abs_diff = (actual - expected).abs().max().item()

        result = benchmark(
            rmsnorm_cuda,
            x,
            weight,
            args.eps,
            rounds=args.rounds,
            warmup=args.warmup,
            iters=args.iters,
        )

        num_elements = batch_size * hidden_size
        approx_bytes = estimate_rmsnorm_bytes(batch_size, hidden_size, dtype)
        approx_gbps = format_gbps(approx_bytes, result["cuda_min_ms"])
        block_size = rmsnorm_cuda_block_size()

        print(
            f"{batch_size:8d} "
            f"{hidden_size:8d} "
            f"{block_size:8d} "
            f"{num_elements:12d} "
            f"{str(dtype):>14} "
            f"{result['wall_min_ms']:14.6f} "
            f"{result['wall_median_ms']:14.6f} "
            f"{result['cuda_min_ms']:14.6f} "
            f"{result['cuda_median_ms']:14.6f} "
            f"{approx_gbps:14.2f} "
            f"{max_abs_diff:14.6e}"
        )


def run_profile(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = parse_dtype(args.dtype)

    print_environment()
    print_kernel_config()

    x = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    print("== Profile Mode ==")
    print(f"batch_size: {args.batch_size}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"block_size: {rmsnorm_cuda_block_size()}")
    print(f"dtype: {dtype}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print()

    for _ in range(args.warmup):
        rmsnorm_cuda(x, weight, args.eps)

    torch.cuda.synchronize()

    range_name = (
        f"rmsnorm_cuda_profile:"
        f"B={args.batch_size},H={args.hidden_size},dtype={args.dtype},iters={args.iters}"
    )

    torch.cuda.nvtx.range_push(range_name)
    try:
        for _ in range(args.iters):
            rmsnorm_cuda(x, weight, args.eps)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()

    print("done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["benchmark", "profile"],
        default="benchmark",
        help="benchmark: stable latency measurement; profile: small workload for nsys",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="fp32", choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.set_grad_enabled(False)

    args = parse_args()

    if args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "profile":
        run_profile(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
