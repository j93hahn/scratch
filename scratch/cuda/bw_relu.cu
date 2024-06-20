#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t x) {
    return x > 0 ? x : 0;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t relu_grad(scalar_t x) {
    return x > 0 ? 1 : 0;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_relu_square(scalar_t x) {
    return x > 0 ? x * x : 0;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_relu_square_grad(scalar_t x) {
    return x > 0 ? 2 * x : 0;
}


template <typename scalar_t>
__global__ void second_forward_cuda_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        scalar_t* __restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx], y = 0; // store global variable indices in registers for faster access
        #pragma unroll 7
        for (int i = 0; i < 7; i++) {
            y += ms[i] * relu(x + bs[i]);
        }
        output[idx] = y;
    }
}


template <typename scalar_t, bool compute_grad_omega>
__global__ void second_backward_cuda_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const int size,
        const scalar_t omega,
        const scalar_t epsilon, // prevent division by zero
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_omega) {
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_grad_omega_raw[];
    scalar_t* shared_grad_omega = reinterpret_cast<scalar_t*>(shared_grad_omega_raw);
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t local_grad_omega = 0;
    if (idx < size) {
        scalar_t x = input[idx], dy = grad_output[idx], dx = 0;
        for (int i = 0; i < 7; i++) {
            scalar_t adjoint = dy * ms[i] * relu_grad(x + bs[i]);
            dx += adjoint * omega;
            if (compute_grad_omega) {
                local_grad_omega += adjoint * (x - 1.5) / (omega + epsilon);
            }
        }
        grad_input[idx] = dx;
    }

    if (!compute_grad_omega) {  // early return if we don't need to compute grad_omega
        return;
    }

    /* notes on further optimizing performance:
        - use warp shuffle instructions to reduce shared memory bank conflicts such as __shfl_down_sync
        - use cooperative groups to reduce warp divergence
        - use dynamic parallelism to launch a separate kernel for grad_omega
        - use library functions such as cub::BlockReduce to perform block-wide reduction

    for now, we use a simple block-wide reduction in shared memory */

    shared_grad_omega[tid] = local_grad_omega;      // initialize shared memory with local gradients
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
__global__ void third_forward_cuda_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const scalar_t* __restrict__ cs,
        const int size,
        scalar_t* __restrict__ output) {
    // create shared memory buffers for very fast memory access
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_data_raw[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_data_raw);
    scalar_t *shared_ms = &shared_data[0];
    scalar_t *shared_bs = &shared_data[4];
    scalar_t *shared_cs = &shared_data[15];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // load ms, bs, and cs into shared memory
        if (threadIdx.x < 4) {
            shared_ms[threadIdx.x] = ms[threadIdx.x];
        }
        if (threadIdx.x < 11) {
            shared_bs[threadIdx.x] = bs[threadIdx.x];
        }
        if (threadIdx.x < 8) {
            shared_cs[threadIdx.x] = cs[threadIdx.x];
        }
        __syncthreads();

        scalar_t x = input[idx], y = 0;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {           // iterate through ms first
            #pragma unroll 8
            for (int j = i; j < i + 8; j++) {   // iterate through bs next
                y += (shared_ms[i] * fused_relu_square(x + shared_bs[j])) * shared_cs[j - i];
            }
        }
        output[idx] = y;
    }
}


template <typename scalar_t, bool compute_grad_omega>
__global__ void third_backward_cuda_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ ms,
        const scalar_t* __restrict__ bs,
        const scalar_t* __restrict__ cs,
        const int size,
        const scalar_t omega,
        const scalar_t epsilon, // prevent division by zero
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_omega) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_grad_omega_raw[];
    scalar_t* shared_grad_omega = reinterpret_cast<scalar_t*>(shared_grad_omega_raw);
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t local_grad_omega = 0;
    if (idx < size) {
        scalar_t x = input[idx], dy = grad_output[idx], dx = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = i; j < i + 8; j++) {
                scalar_t adjoint = dy * ms[i] * fused_relu_square_grad(x + bs[j]) * cs[j - i];
                dx += adjoint * 2 * omega;
                if (compute_grad_omega) {
                    local_grad_omega += adjoint * (x - 5.0) / (omega + epsilon);
                }
            }
        }
        grad_input[idx] = dx;
    }

    if (!compute_grad_omega) {
        return;
    }

    shared_grad_omega[tid] = local_grad_omega;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_grad_omega[tid] += shared_grad_omega[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(grad_omega, shared_grad_omega[0]);
    }
}


/* second order bspline wavelet approximation */
torch::Tensor second_forward_cuda(
        torch::Tensor input,
        torch::Tensor omega,
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
        second_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input_flattened.data_ptr<scalar_t>(),
            ms.data_ptr<scalar_t>(),
            bs.data_ptr<scalar_t>(),
            size,
            output.data_ptr<scalar_t>());
    }));
    return output.view(shape);
}


std::vector<torch::Tensor> second_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor omega,
        torch::Tensor ms,
        torch::Tensor bs) {
    // leave input as original shape - correspondence with grad_output is preserved
    input = omega.item<float>() * input + 1.5;

    // flatten the output Jacobian and create the grad_input tensor
    torch::Tensor grad_output_flattened = grad_output.view({-1});
    torch::Tensor grad_input = torch::zeros_like(grad_output_flattened);

    // create gradient tensors for omega
    torch::Tensor grad_omega = omega.requires_grad() ? torch::zeros_like(omega) : torch::Tensor();

    // launch the kernel
    const auto shape = input.sizes();
    const int size = grad_output_flattened.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    if (omega.requires_grad()) {
        // reduce warp divergence by using a template parameter to select the correct kernel
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_cuda_grad_omega", ([&] {
            second_backward_cuda_kernel<scalar_t, true><<<blocks, threads, threads * sizeof(scalar_t)>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                size,
                omega.item<scalar_t>(),
                1e-8,
                grad_input.data_ptr<scalar_t>(),
                grad_omega.data_ptr<scalar_t>());
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "backward_cuda_no_grad_omega", ([&] {
            second_backward_cuda_kernel<scalar_t, false><<<blocks, threads, 0>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                size,
                omega.item<scalar_t>(),
                1e-8,
                grad_input.data_ptr<scalar_t>(),
                nullptr);
        }));
    }

    // return the gradients
    return {grad_input.view(shape), grad_omega,
            torch::Tensor(), torch::Tensor()};  // ms and bs are hardcoded, no gradients for them
}


/* third order bspline wavelet approximation */
torch::Tensor third_forward_cuda(
        torch::Tensor input,
        torch::Tensor omega,
        torch::Tensor ms,
        torch::Tensor bs,
        torch::Tensor cs) {
    // center the input distribution around x=0 and scale it by omega
    input = (omega.item<float>() * input + 2.5) * 2;

    // flatten the input and create the output tensor
    torch::Tensor input_flattened = input.view({-1});
    torch::Tensor output = torch::zeros_like(input_flattened);

    // launch the kernel and return
    const auto shape = input.sizes();
    const int size = input_flattened.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "third_forward_cuda", ([&] {
        const int shared_data_size = 23 * sizeof(scalar_t);   // 4 + 11 + 8 = 23
        third_forward_cuda_kernel<scalar_t><<<blocks, threads, shared_data_size>>>(
            input_flattened.data_ptr<scalar_t>(),
            ms.data_ptr<scalar_t>(),
            bs.data_ptr<scalar_t>(),
            cs.data_ptr<scalar_t>(),
            size,
            output.data_ptr<scalar_t>());
    }));
    return output.view(shape);
}


std::vector<torch::Tensor> third_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor omega,
        torch::Tensor ms,
        torch::Tensor bs,
        torch::Tensor cs) {
    // leave input as original shape - correspondence with grad_output is preserved
    input = (omega.item<float>() * input + 2.5) * 2;

    // flatten the output Jacobian and create the grad_input tensor
    torch::Tensor grad_output_flattened = grad_output.view({-1});
    torch::Tensor grad_input = torch::zeros_like(grad_output_flattened);

    // create gradient tensors for omega
    torch::Tensor grad_omega = omega.requires_grad() ? torch::zeros_like(omega) : torch::Tensor();

    // launch the kernel
    const auto shape = input.sizes();
    const int size = grad_output_flattened.size(0);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    if (omega.requires_grad()) {
        // reduce warp divergence by using a template parameter to select the correct kernel
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "third_backward_cuda_grad_omega", ([&] {
            third_backward_cuda_kernel<scalar_t, true><<<blocks, threads, threads * sizeof(scalar_t)>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                cs.data_ptr<scalar_t>(),
                size,
                omega.item<scalar_t>(),
                1e-8,
                grad_input.data_ptr<scalar_t>(),
                grad_omega.data_ptr<scalar_t>());
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "third_backward_cuda_no_grad_omega", ([&] {
            third_backward_cuda_kernel<scalar_t, false><<<blocks, threads, 0>>>(
                grad_output_flattened.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                ms.data_ptr<scalar_t>(),
                bs.data_ptr<scalar_t>(),
                cs.data_ptr<scalar_t>(),
                size,
                omega.item<scalar_t>(),
                1e-8,
                grad_input.data_ptr<scalar_t>(),
                nullptr);
        }));
    }

    // return the gradients
    return {grad_input.view(shape), grad_omega,
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
