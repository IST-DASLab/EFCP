# when importing torch, the lib10.so gets loaded into the kernel
# the lib10.so is needed for hinv_cuda, otherwise it will fail
import torch

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='hinv_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'hinv_cuda', ['hinv_cuda.cpp', 'hinv_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
