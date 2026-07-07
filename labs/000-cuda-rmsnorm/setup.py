from setuptools import setup
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
