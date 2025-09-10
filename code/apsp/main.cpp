#include "main.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }

    int m, n;
    infile >> m >> n;
    std::vector<int> edges(n * 3);
    for (int i = 0; i < n * 3; ++i)
        infile >> edges[i];
    infile.close();

    std::vector<int> output(m * m);
    solve(edges.data(), output.data(), m, n);

    // 输出结果
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << output[i * m + j];
            if (j != m - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
