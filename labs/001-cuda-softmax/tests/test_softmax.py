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

import softmax_lab as lab  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_kernel_config() -> None:
    if not lab.cuda_extension_available():
        pytest.skip("Softmax CUDA extension is not built")

    assert lab.block_size() > 0


@pytest.mark.parametrize("variant_name", lab.VARIANTS.keys())
@pytest.mark.parametrize("batch_size,hidden_size", lab.SHAPES)
def test_softmax_matches_torch(
    variant_name: str,
    batch_size: int,
    hidden_size: int,
) -> None:
    variant = lab.VARIANTS[variant_name]
    try:
        lab.check_variant_available(variant)
    except RuntimeError as exc:
        message = str(exc)
        if "not installed" in message or "not built" in message:
            pytest.skip(message)
        raise

    x = lab.make_inputs(
        batch_size,
        hidden_size,
        dtype=torch.float32,
        seed=0,
    )

    lab.assert_close_to_reference(variant, x)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve())]))
