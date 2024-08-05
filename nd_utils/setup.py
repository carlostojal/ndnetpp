from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nd_utils',
    ext_modules=[
        CUDAExtension('nd_utils_cuda', [
            'voxelize.cpp',
            'voxelize_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
