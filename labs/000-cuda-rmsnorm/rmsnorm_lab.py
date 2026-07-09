from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parents[1]

for path in (ROOT, LAB_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

try:
    import _rmsnorm_cuda
except ModuleNotFoundError:
    _rmsnorm_cuda = None


SHAPES: tuple[tuple[int, int], ...] = (
    (1, 1024),
    (1, 4096),
    (8, 4096),
    (32, 4096),
    (32, 8192),
    (128, 8192),
    (256, 8192),
    (512, 8192),
    (1024, 8192),
)

DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Variant:
    name: str
    fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
    backend: str
    dtypes: tuple[str, ...] = ("fp32", "float32")


def cuda_extension_available() -> bool:
    return _rmsnorm_cuda is not None


def require_cuda_extension() -> None:
    if _rmsnorm_cuda is None:
        raise RuntimeError(
            "RMSNorm CUDA extension is not built. Run "
            "`cd labs/000-cuda-rmsnorm && python3 setup.py build_ext --inplace`."
        )


def block_size() -> int:
    require_cuda_extension()
    return int(_rmsnorm_cuda.block_size())


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {name}")
    return DTYPE_MAP[name]


def rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def rmsnorm_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    require_cuda_extension()
    return _rmsnorm_cuda.forward(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_vectorized(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    require_cuda_extension()
    return _rmsnorm_cuda.forward_vectorized(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_warp(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    require_cuda_extension()
    return _rmsnorm_cuda.forward_warp(x.contiguous(), weight.contiguous(), eps)


def rmsnorm_vectorized_warp(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    require_cuda_extension()
    return _rmsnorm_cuda.forward_vectorized_warp(x.contiguous(), weight.contiguous(), eps)


VARIANTS: dict[str, Variant] = {
    "torch": Variant(
        name="torch",
        fn=rmsnorm_torch,
        backend="torch",
        dtypes=("fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
    ),
    "scalar": Variant(name="scalar", fn=rmsnorm_scalar, backend="cuda_extension"),
    "vectorized": Variant(name="vectorized", fn=rmsnorm_vectorized, backend="cuda_extension"),
    "warp": Variant(name="warp", fn=rmsnorm_warp, backend="cuda_extension"),
    "vectorized_warp": Variant(
        name="vectorized_warp",
        fn=rmsnorm_vectorized_warp,
        backend="cuda_extension",
    ),
}

CUDA_EXTENSION_VARIANTS: tuple[str, ...] = (
    "scalar",
    "vectorized",
    "warp",
    "vectorized_warp",
)


def select_variants(spec: str) -> list[Variant]:
    if spec == "all":
        return list(VARIANTS.values())

    selected = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in VARIANTS:
            choices = ", ".join(("all", *VARIANTS.keys()))
            raise ValueError(f"unknown variant: {name}. choices: {choices}")
        selected.append(VARIANTS[name])

    if not selected:
        raise ValueError("at least one variant is required")

    return selected


def check_variant_dtype(variant: Variant, dtype_name: str) -> None:
    if dtype_name not in variant.dtypes:
        supported = ", ".join(variant.dtypes)
        raise ValueError(f"{variant.name} does not support dtype {dtype_name}; supported: {supported}")


def check_variant_available(variant: Variant) -> None:
    if variant.backend == "cuda_extension":
        require_cuda_extension()


def make_inputs(
    batch_size: int,
    hidden_size: int,
    *,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    return x, weight


def assert_close_to_reference(
    variant: Variant,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    expected = rmsnorm_torch(x, weight, eps)
    actual = variant.fn(x, weight, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    return actual


def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def estimate_rmsnorm_bytes(batch_size: int, hidden_size: int, dtype: torch.dtype) -> int:
    # Lower-bound traffic for fused RMSNorm: x read twice, weight read, y write.
    element_size = torch.empty((), dtype=dtype).element_size()
    return 4 * batch_size * hidden_size * element_size
