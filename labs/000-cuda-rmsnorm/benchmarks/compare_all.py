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


def rmsnorm_float4(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_vectorized(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_warp(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_warp(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_float4_warp(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rmsnorm_cuda.forward_vectorized_warp(x.contiguous(), weight.contiguous(), eps)


VARIANTS = {
    "torch": rmsnorm_torch,
    "scalar": rmsnorm_scalar,
    "float4": rmsnorm_float4,
    "warp": rmsnorm_warp,
    "float4_warp": rmsnorm_float4_warp,
}


def benchmark_cuda_event(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iters):
        fn(*args)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def benchmark(fn, *args, rounds: int, warmup: int, iters: int) -> dict[str, float]:
    times = [
        benchmark_cuda_event(fn, *args, warmup=warmup, iters=iters)
        for _ in range(rounds)
    ]

    return {
        "min_ms": min(times),
        "median_ms": statistics.median(times),
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

    print("== RMSNorm Final Comparison ==")
    print(
        f"{'batch':>8} "
        f"{'hidden':>8} "
        f"{'torch_med':>10} "
        f"{'scalar_med':>11} "
        f"{'float4_med':>11} "
        f"{'warp_med':>10} "
        f"{'f4_warp_med':>12} "
        f"{'best':>12} "
        f"{'best_vs_torch':>14} "
        f"{'best_vs_scalar':>15} "
        f"{'max_abs_diff':>14}"
    )

    for batch_size, hidden_size in shapes:
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)

        expected = rmsnorm_torch(x, weight, args.eps)

        outputs = {
            "scalar": rmsnorm_scalar(x, weight, args.eps),
            "float4": rmsnorm_float4(x, weight, args.eps),
            "warp": rmsnorm_warp(x, weight, args.eps),
            "float4_warp": rmsnorm_float4_warp(x, weight, args.eps),
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

        med = {name: value["median_ms"] for name, value in results.items()}

        best_name = min(med, key=med.get)
        best_med = med[best_name]

        best_vs_torch = med["torch"] / best_med
        best_vs_scalar = med["scalar"] / best_med

        print(
            f"{batch_size:8d} "
            f"{hidden_size:8d} "
            f"{med['torch']:10.6f} "
            f"{med['scalar']:11.6f} "
            f"{med['float4']:11.6f} "
            f"{med['warp']:10.6f} "
            f"{med['float4_warp']:12.6f} "
            f"{best_name:>12} "
            f"{best_vs_torch:14.3f} "
            f"{best_vs_scalar:15.3f} "
            f"{max_abs_diff:14.6e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["benchmark"], default="benchmark")

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
    run_benchmark(args)


if __name__ == "__main__":
    main()
