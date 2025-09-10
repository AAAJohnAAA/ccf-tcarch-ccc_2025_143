#include "main.h"

#define INF 1073741823

extern "C" void solve(const int* input_edges, int* output, int m, int n) {
    // 1. 初始化距离矩阵
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i == j)
                output[i * m + j] = 0;
            else
                output[i * m + j] = INF;
        }
    }

    // 2. 将输入边写入距离矩阵
    for (int e = 0; e < n; ++e) {
        int u = input_edges[e * 3 + 0];
        int v = input_edges[e * 3 + 1];
        int w = input_edges[e * 3 + 2];
        output[u * m + v] = w;
    }

    // 3. Floyd-Warshall 主循环
    for (int k = 0; k < m; ++k) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                int ik = output[i * m + k];
                int kj = output[k * m + j];
                if (ik != INF && kj != INF) {
                    int new_dist = ik + kj;
                    if (new_dist < output[i * m + j]) {
                        output[i * m + j] = new_dist;
                    }
                }
            }
        }
    }
}
