from __future__ import annotations

import argparse
import statistics
import time

import torch


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {name}")
    return DTYPE_MAP[name]


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [batch_size, hidden_size]
    # weight: [hidden_size]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def benchmark_wall_time(
    fn,
    *args,
    warmup: int,
    iters: int,
) -> float:
    # End-to-end time observed from Python.
    # Includes Python loop, PyTorch eager dispatch, CUDA kernel launch,
    # GPU execution, and final synchronization.
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
    fn,
    *args,
    warmup: int,
    iters: int,
) -> float:
    # CUDA stream elapsed time.
    # This is closer to GPU-side execution time than wall-clock timing,
    # but it still covers all GPU kernels launched between the two events.
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


def benchmark(
    fn,
    *args,
    rounds: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
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
    # Rough lower-bound estimate only.
    #
    # x read for square/reduction:        B * H
    # x read again for normalization:     B * H
    # weight read:                        B * H
    # output write:                       B * H
    #
    # Total approximate tensor traffic: 4 * B * H * element_size
    #
    # This underestimates PyTorch eager traffic because eager mode creates
    # intermediate tensors such as x.pow(2), rsqrt result, x_norm, etc.
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


def run_benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = parse_dtype(args.dtype)

    print_environment()

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

    print("== RMSNorm PyTorch Eager Baseline ==")
    print(
        f"{'batch':>8} "
        f"{'hidden':>8} "
        f"{'elements':>12} "
        f"{'dtype':>14} "
        f"{'wall_min_ms':>14} "
        f"{'wall_med_ms':>14} "
        f"{'cuda_min_ms':>14} "
        f"{'cuda_med_ms':>14} "
        f"{'approx_GB/s':>14}"
    )

    for batch_size, hidden_size in shapes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)

        result = benchmark(
            rmsnorm_torch,
            x,
            weight,
            rounds=args.rounds,
            warmup=args.warmup,
            iters=args.iters,
        )

        num_elements = batch_size * hidden_size
        approx_bytes = estimate_rmsnorm_bytes(batch_size, hidden_size, dtype)
        approx_gbps = format_gbps(approx_bytes, result["cuda_min_ms"])

        print(
            f"{batch_size:8d} "
            f"{hidden_size:8d} "
            f"{num_elements:12d} "
            f"{str(dtype):>14} "
            f"{result['wall_min_ms']:14.6f} "
            f"{result['wall_median_ms']:14.6f} "
            f"{result['cuda_min_ms']:14.6f} "
            f"{result['cuda_median_ms']:14.6f} "
            f"{approx_gbps:14.2f}"
        )


def run_profile(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = parse_dtype(args.dtype)

    print_environment()

    x = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    print("== Profile Mode ==")
    print(f"batch_size: {args.batch_size}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"dtype: {dtype}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print()

    for _ in range(args.warmup):
        rmsnorm_torch(x, weight)

    torch.cuda.synchronize()

    range_name = (
        f"rmsnorm_torch_profile:"
        f"B={args.batch_size},H={args.hidden_size},dtype={args.dtype},iters={args.iters}"
    )

    torch.cuda.nvtx.range_push(range_name)
    try:
        for _ in range(args.iters):
            rmsnorm_torch(x, weight)

        # Keep synchronization inside the NVTX range so the range includes
        # the submitted GPU work completion from the CPU timeline perspective.
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
