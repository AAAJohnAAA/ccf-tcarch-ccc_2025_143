#include "main.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

// Tunable parameters (experiment with these on MI100)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256            // threads per block (should be multiple of wavefront=64)
#endif
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 4        // each thread processes this many contiguous elements
#endif

// Helper macro for checking HIP errors (debug builds)
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: '%s' (code %d)\n", __FILE__, __LINE__, hipGetErrorName(err), (int)err); \
    } \
} while(0)

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    // device pointers
    int *d_in = nullptr;
    int *d_out = nullptr;
    long long *d_block_sums = nullptr; // use 64-bit for accumulation safety

    const int elems_per_block = BLOCK_SIZE * ELEMS_PER_THREAD;
    const int num_blocks = (N + elems_per_block - 1) / elems_per_block;

    // allocate device memory
    HIP_CHECK( hipMalloc(&d_in, sizeof(int) * (size_t)N) );
    HIP_CHECK( hipMalloc(&d_out, sizeof(int) * (size_t)N) );
    HIP_CHECK( hipMalloc(&d_block_sums, sizeof(long long) * (size_t)num_blocks) );

    // copy input to device
    HIP_CHECK( hipMemcpy(d_in, input, sizeof(int) * (size_t)N, hipMemcpyHostToDevice) );

    // Kernel A: per-block local scan; each thread handles ELEMS_PER_THREAD consecutive elements,
    // computes local prefix and writes to d_out; per-block sum is written to d_block_sums.
    // We'll launch num_blocks blocks with BLOCK_SIZE threads.

    // Define kernel here (lambda style not supported - use global below)
    // Launch block_scan kernel:
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    // compute shared memory: we need BLOCK_SIZE elements for thread_sums (int64)
    size_t shared_bytes = sizeof(long long) * (size_t)BLOCK_SIZE;

    // kernel prototypes (defined below)
    extern __global__ void block_scan_kernel(const int* d_in, int* d_out, long long* d_block_sums, int N);
    hipLaunchKernelGGL(block_scan_kernel, grid, block, shared_bytes, 0, d_in, d_out, d_block_sums, N);
    HIP_CHECK( hipGetLastError() );
    HIP_CHECK( hipDeviceSynchronize() );

    // Copy block sums back to host, do CPU scan (prefix) to get block offsets
    long long* h_block_sums = (long long*) malloc(sizeof(long long) * (size_t)num_blocks);
    long long* h_block_prefix = (long long*) malloc(sizeof(long long) * (size_t)num_blocks);
    if (!h_block_sums || !h_block_prefix) {
        fprintf(stderr, "Host allocation failed\n");
        // cleanup
        if (d_in) hipFree(d_in);
        if (d_out) hipFree(d_out);
        if (d_block_sums) hipFree(d_block_sums);
        if (h_block_sums) free(h_block_sums);
        if (h_block_prefix) free(h_block_prefix);
        return;
    }

    HIP_CHECK( hipMemcpy(h_block_sums, d_block_sums, sizeof(long long) * (size_t)num_blocks, hipMemcpyDeviceToHost) );

    // CPU prefix-sum (exclusive prefix for offsets)
    long long running = 0;
    for (int i = 0; i < num_blocks; ++i) {
        h_block_prefix[i] = running;
        running += h_block_sums[i];
    }

    // Copy block prefix back to device (reuse d_block_sums memory to hold prefixes)
    HIP_CHECK( hipMemcpy(d_block_sums, h_block_prefix, sizeof(long long) * (size_t)num_blocks, hipMemcpyHostToDevice) );

    // Kernel C: add offsets (block_prefix) to each block's outputs
    extern __global__ void add_block_offsets_kernel(int* d_out, const long long* d_block_prefix, int N);
    hipLaunchKernelGGL(add_block_offsets_kernel, grid, block, 0, 0, d_out, d_block_sums, N);
    HIP_CHECK( hipGetLastError() );
    HIP_CHECK( hipDeviceSynchronize() );

    // Copy result back to host output array (int)
    HIP_CHECK( hipMemcpy(output, d_out, sizeof(int) * (size_t)N, hipMemcpyDeviceToHost) );

    // cleanup
    free(h_block_sums);
    free(h_block_prefix);
    HIP_CHECK( hipFree(d_in) );
    HIP_CHECK( hipFree(d_out) );
    HIP_CHECK( hipFree(d_block_sums) );
}

/* ============================
   Device kernels implementation
   ============================ */

/*
  block_scan_kernel:
  - Each thread processes ELEMS_PER_THREAD contiguous elements.
  - Locally compute prefix within thread (in registers).
  - Each thread writes its last local element (thread_sum) into shared memory.
  - Perform parallel exclusive scan over thread_sums in shared memory (size = BLOCK_SIZE).
  - Add scanned thread offset to each local prefix and write to global output.
  - Last thread in block writes the block total to d_block_sums[blockIdx.x].
*/
__global__ void block_scan_kernel(const int* d_in, int* d_out, long long* d_block_sums, int N) {
    // Shared memory for thread sums (one entry per thread)
    extern __shared__ long long s_thread_sums[]; // size = BLOCK_SIZE

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_start = bid * (BLOCK_SIZE * ELEMS_PER_THREAD);
    const int global_thread_base = block_start + tid * ELEMS_PER_THREAD;

    // local buffer in registers
    int vals[ELEMS_PER_THREAD];

    // load values (with bounds check)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int idx = global_thread_base + e;
        vals[e] = (idx < N) ? d_in[idx] : 0;
    }

    // compute local inclusive prefix for this thread
    #pragma unroll
    for (int e = 1; e < ELEMS_PER_THREAD; ++e) {
        vals[e] += vals[e-1];
    }

    // thread's sum (last element)
    long long thread_sum = (long long) vals[ELEMS_PER_THREAD - 1];

    // store thread_sum into shared memory
    s_thread_sums[tid] = thread_sum;
    __syncthreads();

    // Now perform exclusive scan on s_thread_sums (size = BLOCK_SIZE) in-place.
    // We'll use Blelloch scan (upsweep then downsweep) on shared memory.
    // Note: BLOCK_SIZE is compile-time constant and should be a power of two for correctness.
    // If not power-of-two, we handle by padding invisible threads (they hold zero).

    // Upsweep / reduce
    int offset = 1;
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1) - 1;
            int bi = offset*(2*tid+2) - 1;
            // bounds are safe because s_thread_sums size = BLOCK_SIZE
            s_thread_sums[bi] += s_thread_sums[ai];
        }
        offset <<= 1;
    }

    // clear last element for downsweep exclusive scan
    if (tid == 0) {
        long long total = s_thread_sums[BLOCK_SIZE - 1];
        s_thread_sums[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Downsweep
    for (int d = 1; d < BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1) - 1;
            int bi = offset*(2*tid+2) - 1;
            long long t = s_thread_sums[ai];
            s_thread_sums[ai] = s_thread_sums[bi];
            s_thread_sums[bi] += t;
        }
    }
    __syncthreads();

    // Now s_thread_sums[tid] contains exclusive prefix of thread sums (offset to add)
    long long my_offset = s_thread_sums[tid];

    // write out local values + offset to global output
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int idx = global_thread_base + e;
        if (idx < N) {
            long long outv = (long long)vals[e] + my_offset;
            d_out[idx] = (int) outv; // cast back to int; assume no overflow beyond int range
        }
    }

    // thread 0 in block computes block total and writes to d_block_sums; but since we've done
    // reduction, compute block total as sum of (exclusive prefix at last idx + thread_sum_last)
    // easiest: thread with tid == BLOCK_SIZE - 1 computes:
    if (tid == BLOCK_SIZE - 1) {
        // exclusive prefix at last thread is s_thread_sums[tid]
        long long block_total = s_thread_sums[tid] + thread_sum;
        // but note: for correctness if some threads were paddings (beyond last valid thread),
        // their thread_sum was 0 so block_total is fine.
        d_block_sums[bid] = block_total;
    }
}

/*
  add_block_offsets_kernel:
  - For each block, read block prefix (exclusive) from d_block_prefix[blockIdx].
  - Each thread adds that offset to its block elements in d_out (which currently holds local prefixes).
*/
__global__ void add_block_offsets_kernel(int* d_out, const long long* d_block_prefix, int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_start = bid * (BLOCK_SIZE * ELEMS_PER_THREAD);
    const int global_thread_base = block_start + tid * ELEMS_PER_THREAD;

    long long offset = d_block_prefix[bid];

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int idx = global_thread_base + e;
        if (idx < N) {
            long long v = (long long)d_out[idx] + offset;
            d_out[idx] = (int) v;
        }
    }
}
