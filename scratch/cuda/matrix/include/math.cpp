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


// torch::Tensor linear_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(w);
//     CHECK_CUDA(b);
//     assert(x.size(1) == w.size(1));
//     assert(w.size(0) == b.size(0));
//     return matmul(x, w) + b;
// }

// torch::Tensor linear_backward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor grad) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(w);
//     CHECK_CUDA(b);
//     CHECK_CUDA(grad);
//     assert(x.size(1) == w.size(1));
//     assert(w.size(0) == b.size(0));
//     assert(grad.size(0) == b.size(0));
//     assert(grad.size(1) == w.size(0));
//     auto grad_x = matmul(grad, w.t());
//     auto grad_w = matmul(x.t(), grad);
//     auto grad_b = grad.sum(0);
//     return std::make_tuple(grad_x, grad_w, grad_b);
// }


// torch::Tensor relu_forward(torch::Tensor x) {
//     CHECK_CUDA(x);
//     return x.clamp_min(0);
// }


// torch::Tensor relu_backward(torch::Tensor x, torch::Tensor grad) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(grad);
//     return grad * (x > 0).to(x.dtype());
// }


// torch::Tensor softmaxloss_forward(torch::Tensor x, torch::Tensor y) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(y);
//     assert(x.size(0) == y.size(0));
//     assert(x.size(1) == y.size(1));
//     auto x_max = x.max(1, true).values;
//     auto x_exp = (x - x_max).exp();
//     auto x_sum = x_exp.sum(1, true);
//     auto x_log = (x_exp / x_sum).log();
//     auto x_nll = -x_log.gather(1, y.view({-1, 1}));
//     return x_nll.mean();
// }


// torch::Tensor softmaxloss_backward(torch::Tensor x, torch::Tensor y) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(y);
//     assert(x.size(0) == y.size(0));
//     assert(x.size(1) == y.size(1));
//     auto x_max = x.max(1, true).values;
//     auto x_exp = (x - x_max).exp();
//     auto x_sum = x_exp.sum(1, true);
//     auto x_log = (x_exp / x_sum).log();
//     auto x_softmax = x_exp / x_sum;
//     auto x_grad = x_softmax;
//     x_grad.scatter_(1, y.view({-1, 1}), 1);
//     return (x_softmax - x_grad) / x.size(0);
// }


// torch::Tensor sgd_step(torch::Tensor x, torch::Tensor dx, float lr) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(dx);
//     assert(x.sizes() == dx.sizes());
//     return x - lr * dx;
// }


// torch::Tensor adam_step(torch::Tensor x, torch::Tensor dx, torch::Tensor m, torch::Tensor v, float lr, float beta1, float beta2, float eps) {
//     CHECK_CUDA(x);
//     CHECK_CUDA(dx);
//     CHECK_CUDA(m);
//     CHECK_CUDA(v);
//     assert(x.sizes() == dx.sizes());
//     assert(x.sizes() == m.sizes());
//     assert(x.sizes() == v.sizes());
//     m = beta1 * m + (1 - beta1) * dx;
//     v = beta2 * v + (1 - beta2) * dx * dx;
//     auto m_hat = m / (1 - beta1);
//     auto v_hat = v / (1 - beta2);
//     return x - lr * m_hat / (v_hat.sqrt() + eps);
// }


// PYBIND11_MODULE macro creates Python wrappers for the C++ and CUDA functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // basic operations
    m.def("sum", &sum, "element-wise sum of two tensors");
    m.def("matmul", &matmul, "matrix multiplication of two tensors");

    // // neural network layers
    // m.def("linear_forward", &linear_forward, "forward pass of linear layer");
    // m.def("linear_backward", &linear_backward, "backward pass of linear layer");
    // m.def("relu_forward", &relu_forward, "forward pass of ReLU activation");
    // m.def("relu_backward", &relu_backward, "backward pass of ReLU activation");
    // m.def("softmaxloss_forward", &softmaxloss_forward, "forward pass of softmax loss");
    // m.def("softmaxloss_backward", &softmaxloss_backward, "backward pass of softmax loss");

    // // optimizers
    // m.def("sgd_step", &sgd_step, "stochastic gradient descent");
    // m.def("adam_step", &adam_step, "adam optimizer");
}
