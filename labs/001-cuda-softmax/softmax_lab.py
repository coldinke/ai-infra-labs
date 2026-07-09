from __future__ import annotations

import importlib.util
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
    import _softmax_cuda
except ModuleNotFoundError:
    _softmax_cuda = None


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
    fn: Callable[[torch.Tensor], torch.Tensor]
    backend: str
    dtypes: tuple[str, ...] = ("fp32", "float32")


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {name}")
    return DTYPE_MAP[name]


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def softmax_manual_eager(x: torch.Tensor) -> torch.Tensor:
    # Stable softmax:
    #   m = max(x)
    #   y_i = exp(x_i - m) / sum(exp(x_j - m))
    row_max = x.max(dim=-1, keepdim=True).values
    shifted = x - row_max
    exp_x = torch.exp(shifted)
    row_sum = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / row_sum


def cuda_extension_available() -> bool:
    return _softmax_cuda is not None


def require_cuda_extension() -> None:
    if _softmax_cuda is None:
        raise RuntimeError(
            "Softmax CUDA extension is not built. Run "
            "`cd labs/001-cuda-softmax && python3 setup.py build_ext --inplace`."
        )


def block_size() -> int:
    require_cuda_extension()
    return int(_softmax_cuda.block_size())


def softmax_cuda_naive(x: torch.Tensor) -> torch.Tensor:
    require_cuda_extension()
    return _softmax_cuda.forward(x.contiguous())


def triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


VARIANTS: dict[str, Variant] = {
    "torch": Variant(
        name="torch",
        fn=softmax_torch,
        backend="torch",
        dtypes=("fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
    ),
    "manual_eager": Variant(
        name="manual_eager",
        fn=softmax_manual_eager,
        backend="torch_eager",
    ),
    "cuda_naive": Variant(
        name="cuda_naive",
        fn=softmax_cuda_naive,
        backend="cuda_extension",
    ),
}

CUDA_EXTENSION_VARIANTS: tuple[str, ...] = ("cuda_naive",)


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
    if variant.backend in {"torch", "torch_eager"}:
        return

    if variant.backend == "cuda_extension":
        require_cuda_extension()
        return

    if variant.backend == "triton":
        if not triton_available():
            raise RuntimeError("Triton is not installed for this Python environment")
        return

    raise RuntimeError(f"unknown backend for variant {variant.name}: {variant.backend}")


def make_inputs(
    batch_size: int,
    hidden_size: int,
    *,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch_size, hidden_size, device=device, dtype=dtype) * 3.0


def assert_close_to_reference(variant: Variant, x: torch.Tensor) -> torch.Tensor:
    expected = softmax_torch(x)
    actual = variant.fn(x)

    if x.dtype in {torch.float16, torch.bfloat16}:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    return actual


def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def estimate_softmax_bytes(batch_size: int, hidden_size: int, dtype: torch.dtype) -> int:
    # Lower-bound traffic for stable row-wise softmax: x read twice, y write once.
    element_size = torch.empty((), dtype=dtype).element_size()
    return 3 * batch_size * hidden_size * element_size
