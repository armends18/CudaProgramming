// =============================================================================
//  01_vector_add.cu
//  Benchmark: element-wise vector addition  C[i] = A[i] + B[i]
//
//  This is the "Hello World" of CUDA. It's MEMORY-BOUND, so the speedup is
//  modest (often only 5–20x) because the GPU spends most of its time waiting
//  on DRAM, not crunching numbers. Still a great first example.
//
//  Build:  nvcc -O3 -arch=sm_60 01_vector_add.cu -o vector_add
//  Run  :  ./vector_add
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "timer.h"

// ---------- CPU version: plain sequential loop ----------
void vecAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) C[i] = A[i] + B[i];
}

// ---------- GPU kernel: one thread per element ----------
__global__ void vecAddGPU(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    print_header("Benchmark 1 — Vector Addition  C[i] = A[i] + B[i]");

    const int N = 1 << 26;                 // 67 million elements
    const size_t bytes = N * sizeof(float);
    printf("  N = %d elements   (%.1f MB per array)\n",
           N, bytes / (1024.0 * 1024.0));

    // ---------- Allocate & initialize host arrays ----------
    std::vector<float> h_A(N), h_B(N), h_C_cpu(N), h_C_gpu(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;
        h_B[i] = static_cast<float>(i) * 0.002f;
    }

    // ---------- CPU run ----------
    CpuTimer ct; ct.start();
    vecAddCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    double cpu_ms = ct.stop_ms();

    // ---------- GPU run ----------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    // Warm-up launch (first launch pays a one-time JIT/init cost; we don't
    // want that in the measurement)
    vecAddGPU<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer gt; gt.start();
    vecAddGPU<<<blocks, threads>>>(d_A, d_B, d_C, N);
    float gpu_ms = gt.stop_ms();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // ---------- Verify ----------
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3f) { ok = false; break; }
    }

    print_results(cpu_ms, gpu_ms, ok);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
