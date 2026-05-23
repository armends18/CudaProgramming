// =============================================================================
//  04_monte_carlo_pi.cu
//  Benchmark: Monte Carlo estimation of pi
//
//  Throw N random points into the unit square [0,1] x [0,1]. The fraction
//  that lands inside the unit quarter-circle (x^2 + y^2 < 1) approaches
//  pi/4 as N grows. Multiply by 4 to get pi.
//
//  This is EMBARRASSINGLY PARALLEL — every dart is independent — so the GPU
//  speedup is huge.  We use cuRAND on-device to generate random numbers
//  directly on the GPU (otherwise PCIe transfer would dominate).
//
//  Build:  nvcc -O3 -arch=sm_60 04_monte_carlo_pi.cu -lcurand -o monte_carlo_pi
//  Run  :  ./monte_carlo_pi
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include "timer.h"
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- CPU version: Mersenne Twister + serial loop ----------
long long piCPU(long long N) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    long long hits = 0;
    for (long long i = 0; i < N; ++i) {
        double x = dist(rng), y = dist(rng);
        if (x*x + y*y < 1.0) ++hits;
    }
    return hits;
}

// ---------- GPU kernel ----------
//
//  Each thread:
//    1. Seeds its own cuRAND state (so threads don't collide)
//    2. Throws THREAD_TRIALS darts
//    3. Counts how many landed inside the quarter-circle
//    4. Atomically adds its count to a single global counter
//
//  atomicAdd serializes only the final add, not the dart-throwing itself.
__global__ void piGPU(unsigned long long* hits, long long trialsPerThread,
                      unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread gets its own RNG state, seeded uniquely
    curandState state;
    curand_init(seed, tid, 0, &state);

    unsigned long long local = 0;
    for (long long i = 0; i < trialsPerThread; ++i) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x*x + y*y < 1.f) ++local;
    }
    atomicAdd(hits, local);   // one atomic per thread, not per dart
}

int main() {
    print_header("Benchmark 4 — Monte Carlo Estimation of pi");

    const long long N_CPU = 100'000'000LL;          // 100 M darts for CPU

    // GPU runs many more because it can; we still scale CPU/GPU times to a
    // common "throughput" so the comparison is fair.
    const int  threads          = 256;
    const int  blocks           = 4096;
    const long long perThread   = 10000;
    const long long N_GPU       = (long long)threads * blocks * perThread;

    printf("  CPU darts   : %lld\n", N_CPU);
    printf("  GPU darts   : %lld   (%d blocks x %d threads x %lld each)\n",
           N_GPU, blocks, threads, perThread);

    // ---------- CPU run ----------
    printf("  Running CPU...\n");
    CpuTimer ct; ct.start();
    long long cpu_hits = piCPU(N_CPU);
    double cpu_ms = ct.stop_ms();
    double pi_cpu = 4.0 * (double)cpu_hits / (double)N_CPU;

    // ---------- GPU run ----------
    unsigned long long* d_hits;
    CUDA_CHECK(cudaMalloc(&d_hits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    // Warm-up
    piGPU<<<blocks, threads>>>(d_hits, 1, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    GpuTimer gt; gt.start();
    piGPU<<<blocks, threads>>>(d_hits, perThread, 42ULL);
    float gpu_ms = gt.stop_ms();
    CUDA_CHECK(cudaGetLastError());

    unsigned long long gpu_hits = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_hits, d_hits, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    double pi_gpu = 4.0 * (double)gpu_hits / (double)N_GPU;

    // ---------- Throughput-normalized comparison ----------
    // ms per million darts — lower is better
    double cpu_per_M = cpu_ms / (N_CPU / 1.0e6);
    double gpu_per_M = gpu_ms / (N_GPU / 1.0e6);

    bool ok = std::fabs(pi_gpu - M_PI) < 0.001 && std::fabs(pi_cpu - M_PI) < 0.001;

    printf("\n");
    printf("  ----------------------------------------------------\n");
    printf("    CPU pi estimate :  %.6f   (true pi = %.6f)\n", pi_cpu, M_PI);
    printf("    GPU pi estimate :  %.6f\n", pi_gpu);
    printf("    CPU time        :  %10.3f ms   (%.3f ms / 1M darts)\n",
           cpu_ms, cpu_per_M);
    printf("    GPU time        :  %10.3f ms   (%.3f ms / 1M darts)\n",
           gpu_ms, gpu_per_M);
    printf("    Throughput x    :  %10.2fx faster per-dart on GPU\n",
           cpu_per_M / gpu_per_M);
    printf("    Correctness     :  %s\n", ok ? "PASS" : "FAIL");
    printf("  ----------------------------------------------------\n\n");

    cudaFree(d_hits);
    return 0;
}
