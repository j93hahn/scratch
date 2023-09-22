#include <torch/extension.h>
#include <cassert>


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")


// CUDA forward declarations
torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);


// C++ interface
torch::Tensor sum(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    assert(a.sizes() == b.sizes());
    return sum_cuda(a, b);
}


torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    assert(a.size(1) == b.size(0));
    return matmul_cuda(a, b);
}


// PYBIND11_MODULE macro creates Python wrappers for the C++ and CUDA functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum", &sum, "element-wise sum of two tensors");
    m.def("matmul", &matmul, "matrix multiplication of two tensors");
}
