#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <fstream>

extern "C" void solve(const int* input_edges, int* output, int m, int n);

#endif
