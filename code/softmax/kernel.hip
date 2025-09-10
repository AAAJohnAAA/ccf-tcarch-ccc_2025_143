#include "main.h"
#include <cmath>
#include <float.h>

// --- GPU 核函数 --- //

// 并行求最大值
__global__ void max_reduce_kernel(const float* d_in, float* d_max, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? d_in[idx] : -FLT_MAX;
    __syncthreads();

    // 并行归约求最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        d_max[blockIdx.x] = sdata[0];
}

// 并行计算 exp(x - max)
__global__ void softmax_exp_kernel(const float* d_in, float* d_out, int N, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = __expf(d_in[idx] - max_val);
    }
}

// 并行求和
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

// 并行归一化
__global__ void normalize_kernel(float* d_out, float sum_exp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] /= sum_exp;
}

// --- Host solve 函数 ---
extern "C" void solve(const float* input, float* output, int N) {
    float *d_in = nullptr, *d_out = nullptr;
    hipError_t err;

    // 分配 GPU 内存
    err = hipMalloc(&d_in, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_in failed\n"; exit(1); }
    err = hipMalloc(&d_out, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_out failed\n"; exit(1); }

    err = hipMemcpy(d_in, input, N * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "hipMemcpy to device failed\n"; exit(1); }

    int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- 求最大值 ---
    float* d_max_blocks = nullptr;
    err = hipMalloc(&d_max_blocks, gridSize * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_max_blocks failed\n"; exit(1); }

    max_reduce_kernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_max_blocks, N);

    float* h_max_blocks = new float[gridSize];
    err = hipMemcpy(h_max_blocks, d_max_blocks, gridSize * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy max failed\n"; exit(1); }

    float max_val = -FLT_MAX;
    for (int i = 0; i < gridSize; ++i) max_val = fmaxf(max_val, h_max_blocks[i]);
    delete[] h_max_blocks;

    // --- 计算 exp(x - max) ---
    softmax_exp_kernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N, max_val);

    // --- 求指数和 ---
    float* d_sum_blocks = nullptr;
    err = hipMalloc(&d_sum_blocks, gridSize * sizeof(float));
    if (err != hipSuccess) { std::cerr << "hipMalloc d_sum_blocks failed\n"; exit(1); }

    sum_reduce_kernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_out, d_sum_blocks, N);

    float* h_sum_blocks = new float[gridSize];
    err = hipMemcpy(h_sum_blocks, d_sum_blocks, gridSize * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy sum failed\n"; exit(1); }

    float sum_exp = 0.0f;
    for (int i = 0; i < gridSize; ++i) sum_exp += h_sum_blocks[i];
    delete[] h_sum_blocks;

    // --- 归一化 ---
    normalize_kernel<<<gridSize, BLOCK_SIZE>>>(d_out, sum_exp, N);

    // --- 拷回结果 ---
    err = hipMemcpy(output, d_out, N * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "hipMemcpy to host failed\n"; exit(1); }

    // --- 释放资源 ---
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_max_blocks);
    hipFree(d_sum_blocks);
}
