from __future__ import annotations

import sys
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

import _rmsnorm_cuda  # noqa: E402


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def check_shape(batch_size: int, hidden_size: int) -> None:
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.float32
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    expected = rmsnorm_torch(x, weight, eps)
    actual = _rmsnorm_cuda.forward_warp(x.contiguous(), weight.contiguous(), eps)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    max_abs_diff = (actual - expected).abs().max().item()
    print(
        f"PASS warp batch_size={batch_size:<4} "
        f"hidden_size={hidden_size:<6} "
        f"max_abs_diff={max_abs_diff:.6e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

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

    for batch_size, hidden_size in shapes:
        check_shape(batch_size, hidden_size)


if __name__ == "__main__":
    main()
