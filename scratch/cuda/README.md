# CUDA Kernels

This directory contains several CUDA C++ kernels that I have written myself.
1] `sum.cu` - a demo taken from this wonderful [tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) provided by NVIDIA. Contains minor comments and modifications designed to help me understand the semantics of CUDA C++ better.
2] `matrix/` - a simple kernel that can compute element-wise addition and matrix multiplication of two matrices. Uses PyTorch's [JIT compiler](https://pytorch.org/docs/stable/generated/torch.jit.load.html#torch.jit.load) to generate a CUDA kernel that can be called from a Python script.
3] `nerf/` - a CUDA kernel that can render a 3D scene using the original [NeRF](https://arxiv.org/abs/2003.08934) ray-tracing algorithm. Built entirely in CUDA and C++.
