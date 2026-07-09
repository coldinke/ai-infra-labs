from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BENCH_DIR = Path(__file__).resolve().parent
LAB_DIR = BENCH_DIR.parent
ROOT = LAB_DIR.parents[1]

for path in (ROOT, LAB_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from labkit.bench import benchmark_cuda, format_gbps  # noqa: E402
from labkit.cuda import print_torch_cuda_environment, require_cuda, run_nvtx_profile  # noqa: E402

import rmsnorm_lab as lab  # noqa: E402


def print_environment() -> None:
    extra = {}
    if lab.cuda_extension_available():
        extra["rmsnorm_cuda_block_size"] = lab.block_size()
    print_torch_cuda_environment(extra)


def run_benchmark(args: argparse.Namespace) -> None:
    dtype = lab.parse_dtype(args.dtype)
    variants = lab.select_variants(args.variant)

    for variant in variants:
        lab.check_variant_dtype(variant, args.dtype)
        lab.check_variant_available(variant)

    print_environment()

    print("== RMSNorm Benchmark ==")
    print(
        f"{'batch':>8} "
        f"{'hidden':>8} "
        f"{'variant':>12} "
        f"{'backend':>16} "
        f"{'dtype':>14} "
        f"{'wall_min_ms':>14} "
        f"{'wall_med_ms':>14} "
        f"{'cuda_min_ms':>14} "
        f"{'cuda_med_ms':>14} "
        f"{'approx_GB/s':>14} "
        f"{'max_abs_diff':>14}"
    )

    for batch_size, hidden_size in lab.SHAPES:
        x, weight = lab.make_inputs(
            batch_size,
            hidden_size,
            dtype=dtype,
            seed=args.seed,
        )
        expected = lab.rmsnorm_torch(x, weight, args.eps)

        for variant in variants:
            actual = variant.fn(x, weight, args.eps)
            if variant.name != "torch":
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

            result = benchmark_cuda(
                variant.fn,
                x,
                weight,
                args.eps,
                rounds=args.rounds,
                warmup=args.warmup,
                iters=args.iters,
            )
            approx_bytes = lab.estimate_rmsnorm_bytes(batch_size, hidden_size, dtype)
            approx_gbps = format_gbps(approx_bytes, result["cuda_min_ms"])
            diff = lab.max_abs_diff(actual, expected)

            print(
                f"{batch_size:8d} "
                f"{hidden_size:8d} "
                f"{variant.name:>12} "
                f"{variant.backend:>16} "
                f"{str(dtype):>14} "
                f"{result['wall_min_ms']:14.6f} "
                f"{result['wall_median_ms']:14.6f} "
                f"{result['cuda_min_ms']:14.6f} "
                f"{result['cuda_median_ms']:14.6f} "
                f"{approx_gbps:14.2f} "
                f"{diff:14.6e}"
            )


def run_profile(args: argparse.Namespace) -> None:
    dtype = lab.parse_dtype(args.dtype)
    variants = lab.select_variants(args.variant)

    if len(variants) != 1:
        raise ValueError("profile mode requires exactly one --variant")

    variant = variants[0]
    lab.check_variant_dtype(variant, args.dtype)
    lab.check_variant_available(variant)

    print_environment()

    x, weight = lab.make_inputs(
        args.batch_size,
        args.hidden_size,
        dtype=dtype,
        seed=args.seed,
    )

    print("== Profile Mode ==")
    print(f"variant: {variant.name}")
    print(f"backend: {variant.backend}")
    print(f"batch_size: {args.batch_size}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"dtype: {dtype}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print()

    range_name = (
        f"rmsnorm_{variant.name}_profile:"
        f"B={args.batch_size},H={args.hidden_size},dtype={args.dtype},iters={args.iters}"
    )
    run_nvtx_profile(
        variant.fn,
        x,
        weight,
        args.eps,
        warmup=args.warmup,
        iters=args.iters,
        range_name=range_name,
    )

    print("done")


def parse_args(
    argv: list[str] | None = None,
    *,
    default_variant: str = "all",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--mode",
        choices=["benchmark", "profile"],
        default="benchmark",
        help="benchmark: stable latency measurement; profile: small workload for nsys",
    )
    parser.add_argument(
        "--variant",
        default=default_variant,
        help="variant name, comma-separated names, or all",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="fp32", choices=sorted(lab.DTYPE_MAP.keys()))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    default_variant: str = "all",
) -> None:
    args = parse_args(argv, default_variant=default_variant)

    require_cuda()
    torch.set_grad_enabled(False)

    if args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "profile":
        run_profile(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
