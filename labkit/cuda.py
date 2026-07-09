from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


def print_torch_cuda_environment(extra: Mapping[str, object] | None = None) -> None:
    print("== Environment ==")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    if extra:
        for key, value in extra.items():
            print(f"{key}: {value}")

    print()


def run_nvtx_profile(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int,
    iters: int,
    range_name: str,
) -> None:
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(range_name)
    try:
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()

