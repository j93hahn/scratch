#include <torch/extension.h>
#include <stdio.h>
#include <vector>


__global__ void sum_cuda_kernel(float *a, float *b, float *c, int num_el) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_el) {
        c[idx] = a[idx] + b[idx];
    }
}


__global__ void matmul_cuda_kernel(float *a, float *b, float *c, int m, int k, int n) {
    /* a, b, c are one-dimensional tensors */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {  // idx indexes into output tensor c; must be less than c.num_el()
        int i = idx / n;    // row
        int j = idx % n;    // column
        float sum = 0.f;
        for (int l = 0; l < k; l++) {
            sum += a[i * k + l] * b[l * n + j];
        }
        c[idx] = sum;
    }
}


torch::Tensor sum_cuda(torch::Tensor a, torch::Tensor b) {
    const int threads = 256;
    const int num_el = a.size(0) * a.size(1);
    const int blocks = (num_el + threads - 1) / threads;

    torch::Tensor c = torch::zeros(a.sizes(), a.options());
    sum_cuda_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), num_el);
    return c;
}


torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int threads = 256;
    const int num_el = a.size(0) * b.size(1);
    const int blocks = (num_el + threads - 1) / threads;

    torch::Tensor c = torch::zeros({a.size(0), b.size(1)}, a.options());
    matmul_cuda_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                                            a.size(0), a.size(1), b.size(1));
    return c;
}
