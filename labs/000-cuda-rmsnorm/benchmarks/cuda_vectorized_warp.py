from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import _rmsnorm_cuda  # noqa: E402


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


def rmsnorm_scalar(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_vectorized(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_vectorized(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_warp(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_warp(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_vectorized_warp(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_vectorized_warp(x.contiguous(), weight.contiguous(), eps)


VARIANTS = {
    "scalar": rmsnorm_scalar,
    "vectorized": rmsnorm_vectorized,
    "warp": rmsnorm_warp,
    "vectorized_warp": rmsnorm_vectorized_warp,
}


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
    times = []

    for _ in range(rounds):
        times.append(benchmark_cuda_event(fn, *args, warmup=warmup, iters=iters))

    return {
        "cuda_min_ms": min(times),
        "cuda_median_ms": statistics.median(times),
    }


def print_environment() -> None:
    print("== Environment ==")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"rmsnorm_cuda_block_size: {_rmsnorm_cuda.block_size()}")
    print()


def run_benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = torch.float32

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

    print("== RMSNorm Variant Comparison ==")
    print(
        f"{'batch':>8} "
        f"{'hidden':>8} "
        f"{'scalar_med':>12} "
        f"{'vec_med':>12} "
        f"{'warp_med':>12} "
        f"{'vec_warp_med':>14} "
        f"{'best':>16} "
        f"{'best_speedup':>14} "
        f"{'max_abs_diff':>14}"
    )

    for batch_size, hidden_size in shapes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)

        expected = rmsnorm_torch(x, weight, args.eps)

        outputs = {
            name: fn(x, weight, args.eps)
            for name, fn in VARIANTS.items()
        }

        for name, out in outputs.items():
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

        max_abs_diff = max(
            (out - expected).abs().max().item()
            for out in outputs.values()
        )

        results = {
            name: benchmark(
                fn,
                x,
                weight,
                args.eps,
                rounds=args.rounds,
                warmup=args.warmup,
                iters=args.iters,
            )
            for name, fn in VARIANTS.items()
        }

        scalar_med = results["scalar"]["cuda_median_ms"]
        vec_med = results["vectorized"]["cuda_median_ms"]
        warp_med = results["warp"]["cuda_median_ms"]
        vec_warp_med = results["vectorized_warp"]["cuda_median_ms"]

        best_name = min(results, key=lambda name: results[name]["cuda_median_ms"])
        best_med = results[best_name]["cuda_median_ms"]
        best_speedup = scalar_med / best_med

        print(
            f"{batch_size:8d} "
            f"{hidden_size:8d} "
            f"{scalar_med:12.6f} "
            f"{vec_med:12.6f} "
            f"{warp_med:12.6f} "
            f"{vec_warp_med:14.6f} "
            f"{best_name:>16} "
            f"{best_speedup:14.3f} "
            f"{max_abs_diff:14.6e}"
        )


def run_profile(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda"
    dtype = torch.float32

    print_environment()

    x = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    fn = VARIANTS[args.variant]

    print("== Profile Mode ==")
    print(f"variant: {args.variant}")
    print(f"batch_size: {args.batch_size}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print()

    for _ in range(args.warmup):
        fn(x, weight, args.eps)

    torch.cuda.synchronize()

    range_name = (
        f"rmsnorm_{args.variant}_profile:"
        f"B={args.batch_size},H={args.hidden_size},iters={args.iters}"
    )

    torch.cuda.nvtx.range_push(range_name)
    try:
        for _ in range(args.iters):
            fn(x, weight, args.eps)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()

    print("done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["benchmark", "profile"], default="benchmark")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        default="vectorized_warp",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8192)

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
