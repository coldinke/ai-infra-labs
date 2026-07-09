from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


TEST_DIR = Path(__file__).resolve().parent
LAB_DIR = TEST_DIR.parent
ROOT = LAB_DIR.parents[1]

for path in (ROOT, LAB_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import rmsnorm_lab as lab  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_kernel_config() -> None:
    if not lab.cuda_extension_available():
        pytest.skip("RMSNorm CUDA extension is not built")

    assert lab.block_size() > 0


@pytest.mark.parametrize("variant_name", lab.CUDA_EXTENSION_VARIANTS)
@pytest.mark.parametrize("batch_size,hidden_size", lab.SHAPES)
def test_rmsnorm_matches_torch(
    variant_name: str,
    batch_size: int,
    hidden_size: int,
) -> None:
    if not lab.cuda_extension_available():
        pytest.skip("RMSNorm CUDA extension is not built")

    variant = lab.VARIANTS[variant_name]
    x, weight = lab.make_inputs(
        batch_size,
        hidden_size,
        dtype=torch.float32,
        seed=0,
    )

    lab.assert_close_to_reference(variant, x, weight, eps=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve())]))
