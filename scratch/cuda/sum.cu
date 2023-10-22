#include <iostream>
#include <stdio.h>
#include <math.h>


// compile this with: nvcc -o sum sum.cu


/* kernel */
__global__  // __global__ signifies a kernel (device code) that can be called from host code (CPU)
void add_cuda(int n, float *x, float *y)
{
    /*
    CUDA provides several built-in variables that can be used to identify the thread
    and its position within the grid of threads launched by the kernel.

    - threadIdx: the index of the thread within its block
    - blockIdx: the index of the block within the grid
    - blockDim: the dimensions of the block (num threads per block)
    - gridDim: the dimensions of the grid (num blocks per grid)
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /* above, stride is the total number of threads in the grid -
    in essence, each thread does one iteration of the for loop below
    */

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    /*
    key insight - the kernel is executed by many threads in parallel

    add<<<x, y>>> is the execution configuration
    - x is the number of thread blocks
    - y is the number of threads in a thread block (multiple of 32)

    <<<1, 256>>> means 1 block of 256 threads

    when calculating numBlocks, we round up to ensure that we have
    enough threads to cover the entire array
    */
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add_cuda<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
