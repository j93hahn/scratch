#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename scalar_t>
__device__ __forceinline__ scalar_t max_scalar(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}


template <typename scalar_t>
__global__ void forward_cuda_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        const scalar_t alpha,
        scalar_t* __restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx], y = 0; // store global variables in registers for faster access
        for (int i = 0; i < 7; i++) {
            y += (ms[i] * max_scalar(x + bs[i], static_cast<scalar_t>(0))) * alpha;
        }
        output[idx] = y;
    }
}


template <typename scalar_t>
__global__ void backward_cuda_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        const scalar_t omega,
        const scalar_t alpha,
        scalar_t* __restrict__ grad_input) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx], dy = grad_output[idx], dx = 0;
        for (int i = 0; i < 7; i++) {
            dx += dy * ms[i] * (x + bs[i] > 0) * omega * alpha;
        }
        grad_input[idx] = dx;
    }
}


template <typename scalar_t>
__global__ void backward_omega_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        const scalar_t omega,
        const scalar_t alpha,
        const scalar_t epsilon, // prevent division by zero
        scalar_t* __restrict__ grad_omega) {
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_grad_omega_raw[];
    scalar_t* shared_grad_omega = reinterpret_cast<scalar_t*>(shared_grad_omega_raw);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t local_grad = 0;
    if (idx < size) {
        scalar_t x = input[idx], dy = grad_output[idx];
        for (int i = 0; i < 7; i++) {
            local_grad += dy * alpha * ms[i] * (x + bs[i] > 0) * (x - 1.5) / (omega + epsilon);
        }
    }
    shared_grad_omega[tid] = local_grad;            // initialize shared memory with local gradients
    __syncthreads();                                // ensure all writes to shared memory are completed

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  // perform block-wide reduction in shared memory
        if (tid < s) {
            shared_grad_omega[tid] += shared_grad_omega[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // atomic add to global gradient by the first thread in the block
        atomicAdd(grad_omega, shared_grad_omega[0]);
    }
}


template <typename scalar_t>
__global__ void backward_alpha_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        scalar_t* __restrict__ grad_alpha) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_grad_alpha_raw[];
    scalar_t* shared_grad_alpha = reinterpret_cast<scalar_t*>(shared_grad_alpha_raw);
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t local_grad = 0;
    if (idx < size) {
        scalar_t x = input[idx], dy = grad_output[idx];
        for (int i = 0; i < 7; i++) {
            local_grad += dy * ms[i] * max_scalar(x + bs[i], static_cast<scalar_t>(0));
        }
    }
    shared_grad_alpha[tid] = local_grad;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_grad_alpha[tid] += shared_grad_alpha[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(grad_alpha, shared_grad_alpha[0]);
    }
}


torch::Tensor forward_cuda(
        torch::Tensor input,
        torch::Tensor omega,
        torch::Tensor alpha,
        torch::Tensor ms,
        torch::Tensor bs) {
    // center the input distribution around x=0 and scale it by omega
    input = omega.item<float>() * input + 1.5;

    // flatten the input and create the output tensor
    torch::Tensor input_flattened = input.view({-1});
    torch::Tensor output = torch::zeros_like(input_flattened);

    // launch the kernel and return
    const auto shape = input.sizes();
    const int size = input_flattened.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", ([&] {
        forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input_flattened.data_ptr<scalar_t>(),
            ms.data_ptr<scalar_t>(),
            bs.data_ptr<scalar_t>(),
            size,
            alpha.item<scalar_t>(),
            output.data_ptr<scalar_t>());
    }));
    return output.view(shape);
}


std::vector<torch::Tensor> backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor omega,
        torch::Tensor alpha,
        torch::Tensor ms,
        torch::Tensor bs) {
    // leave input as original shape - correspondence with grad_output is preserved
    input = omega.item<float>() * input + 1.5;

    // flatten the output Jacobian and create the grad_input tensor
    torch::Tensor grad_output_flattened = grad_output.view({-1});
    torch::Tensor grad_input = torch::zeros_like(grad_output_flattened);

    // launch the kernel
    const auto shape = input.sizes();
    const int size = grad_output_flattened.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_cuda", ([&] {
        backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_flattened.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            ms.data_ptr<scalar_t>(),
            bs.data_ptr<scalar_t>(),
            size,
            omega.item<scalar_t>(),
            alpha.item<scalar_t>(),
            grad_input.data_ptr<scalar_t>());
    }));

    /* check if we need gradients on our hyperparameters and launch separate kernels for them;
        we use shared memory to perform block-wide reduction for the gradients of omega and alpha */
    torch::Tensor grad_omega = torch::Tensor();
    torch::Tensor grad_alpha = torch::Tensor();
    if (omega.requires_grad()) {
        grad_omega = torch::zeros_like(omega);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_omega_kernel", ([&] {
            backward_omega_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                size,
                omega.item<scalar_t>(),
                alpha.item<scalar_t>(),
                1e-8,
                grad_omega.data_ptr<scalar_t>());
        }));
    }
    if (alpha.requires_grad()) {
        grad_alpha = torch::zeros_like(alpha);
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_alpha_kernel", ([&] {
            backward_alpha_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                size,
                grad_alpha.data_ptr<scalar_t>());
        }));
    }

    // return the gradients
    return {grad_input.view(shape), grad_omega, grad_alpha,
            torch::Tensor(), torch::Tensor()};  // ms and bs are hardcoded, no gradients for them
}