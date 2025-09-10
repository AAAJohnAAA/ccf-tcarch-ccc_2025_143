#include "main.h"
#include <float.h>

// 并行求最大值 kernel
__global__ void max_reduce_kernel(const float* d_in, float* d_max, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? d_in[idx] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        d_max[blockIdx.x] = sdata[0];
}

// 并行求和 kernel
__global__ void sum_reduce_kernel(float* d_in, float* d_sum, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_sum[blockIdx.x] = sdata[0];
}

// 计算 exp(x - max) kernel
__global__ void exp_kernel(const float* d_in, float* d_out, int N, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] = __expf(d_in[idx] - max_val);
}

// 归一化 kernel
__global__ void normalize_kernel(float* d_out, int N, float sum_exp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] /= sum_exp;
}

extern "C" void solve(const float* input, float* output, int N) {
    float *d_in = nullptr, *d_out = nullptr;
    hipError_t err;

    // 分配 GPU 内存
    err = hipMalloc(&d_in, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_in failed\n"; exit(1); }

    err = hipMalloc(&d_out, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_out failed\n"; exit(1); }

    // 拷贝输入数据到 GPU
    err = hipMemcpy(d_in, input, N * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy to d_in failed\n"; exit(1); }

    // --- 1. 求最大值 ---
    int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* d_max_blocks = nullptr;
    err = hipMalloc(&d_max_blocks, gridSize * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_max_blocks failed\n"; exit(1); }

    max_reduce_kernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_max_blocks, N);
    err = hipDeviceSynchronize(); if (err != hipSuccess) { std::cerr << "hipDeviceSynchronize failed\n"; exit(1); }

    // CPU 归约 block 最大值
    float* h_max_blocks = new float[gridSize];
    err = hipMemcpy(h_max_blocks, d_max_blocks, gridSize * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy d_max_blocks to host failed\n"; exit(1); }

    float max_val = -FLT_MAX;
    for (int i = 0; i < gridSize; ++i) max_val = fmaxf(max_val, h_max_blocks[i]);
    delete[] h_max_blocks;

    // --- 2. 计算 exp(x - max_val) ---
    exp_kernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N, max_val);
    err = hipDeviceSynchronize(); if (err != hipSuccess) { std::cerr << "exp_kernel failed\n"; exit(1); }

    // --- 3. 求指数和 ---
    float* d_sum_blocks = nullptr;
    err = hipMalloc(&d_sum_blocks, gridSize * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_sum_blocks failed\n"; exit(1); }

    sum_reduce_kernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_out, d_sum_blocks, N);
    err = hipDeviceSynchronize(); if (err != hipSuccess) { std::cerr << "sum_reduce_kernel failed\n"; exit(1); }

    float* h_sum_blocks = new float[gridSize];
    err = hipMemcpy(h_sum_blocks, d_sum_blocks, gridSize * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy d_sum_blocks to host failed\n"; exit(1); }

    float sum_exp = 0.0f;
    for (int i = 0; i < gridSize; ++i) sum_exp += h_sum_blocks[i];
    delete[] h_sum_blocks;

    // --- 4. 归一化 ---
    normalize_kernel<<<gridSize, BLOCK_SIZE>>>(d_out, N, sum_exp);
    err = hipDeviceSynchronize(); if (err != hipSuccess) { std::cerr << "normalize_kernel failed\n"; exit(1); }

    // 拷贝结果回 CPU
    err = hipMemcpy(output, d_out, N * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy output failed\n"; exit(1); }

    // 释放 GPU 内存
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_max_blocks);
    hipFree(d_sum_blocks);
}
