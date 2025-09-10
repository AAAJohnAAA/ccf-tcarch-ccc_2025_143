#include "main.h"

#define INF 1073741823  // 表示不可达
#define BLOCK_SIZE 16   // 每个 block 的线程块尺寸，可调优

// GPU kernel：针对给定中间顶点 k，更新 dist[i][j]
__global__ void floyd_warshall_step(int* dist, int m, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < m) {
        int ik = dist[i * m + k];
        int kj = dist[k * m + j];
        if (ik != INF && kj != INF) {
            int new_dist = ik + kj;
            if (new_dist < dist[i * m + j]) {
                dist[i * m + j] = new_dist;
            }
        }
    }
}

// 主函数接口：在 GPU 上执行 Floyd-Warshall
extern "C" void solve(const int* input_edges, int* output, int m, int n) {
    // 初始化距离矩阵
    std::vector<int> h_dist(m * m, INF);
    for (int i = 0; i < m; ++i)
        h_dist[i * m + i] = 0;

    // 将输入边拷贝到矩阵
    for (int e = 0; e < n; ++e) {
        int u = input_edges[e * 3 + 0];
        int v = input_edges[e * 3 + 1];
        int w = input_edges[e * 3 + 2];
        h_dist[u * m + v] = w;
    }

    // 分配 GPU 内存并拷贝
    int* d_dist;
    hipMalloc(&d_dist, m * m * sizeof(int));
    hipMemcpy(d_dist, h_dist.data(), m * m * sizeof(int), hipMemcpyHostToDevice);

    // 设置线程块和网格尺寸
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // Floyd-Warshall 主循环
    for (int k = 0; k < m; ++k) {
        floyd_warshall_step<<<grid, block>>>(d_dist, m, k);
        hipDeviceSynchronize(); // 等待 GPU 完成
    }

    // 拷贝回 CPU
    hipMemcpy(output, d_dist, m * m * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_dist);
}
