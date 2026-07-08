import glob
import os
import shutil
from pathlib import Path

from setuptools import setup


def configure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return

    nvcc = os.environ.get("NVCC_BIN") or shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        os.environ["CUDA_HOME"] = str(nvcc_path.parent.parent)
        os.environ["PATH"] = f"{nvcc_path.parent}:{os.environ.get('PATH', '')}"
        return

    for pattern in (
        "/usr/local/cuda*/bin/nvcc",
        "/opt/cuda*/bin/nvcc",
        "/opt/nvidia/**/bin/nvcc",
    ):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            nvcc_path = Path(matches[0]).resolve()
            os.environ["CUDA_HOME"] = str(nvcc_path.parent.parent)
            os.environ["PATH"] = f"{nvcc_path.parent}:{os.environ.get('PATH', '')}"
            return


configure_cuda_home()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm_cuda",
    ext_modules=[
        CUDAExtension(
            name="_rmsnorm_cuda",
            sources=[
                "src/rmsnorm.cpp",
                "src/rmsnorm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
