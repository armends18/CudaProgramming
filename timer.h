// timer.h — small helpers for CPU & GPU timing and a common results table.
// Shared by all four benchmarks.
#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// ---------- CPU timer (std::chrono, milliseconds) ----------
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---------- GPU timer (CUDA events, milliseconds) ----------
struct GpuTimer {
    cudaEvent_t e0, e1;
    GpuTimer()  { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    ~GpuTimer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    void start() { cudaEventRecord(e0); }
    float stop_ms() {
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms;
    }
};

// ---------- CUDA error checking ----------
#define CUDA_CHECK(call) do {                                                \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                      \
                cudaGetErrorName(err), __FILE__, __LINE__,                   \
                cudaGetErrorString(err));                                    \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

// ---------- Pretty results banner ----------
inline void print_header(const char* title) {
    printf("\n");
    printf("=============================================================\n");
    printf("  %s\n", title);
    printf("=============================================================\n");
}

inline void print_results(double cpu_ms, double gpu_ms, bool ok) {
    double speedup = cpu_ms / gpu_ms;
    printf("\n");
    printf("  ----------------------------------------------------\n");
    printf("    CPU time       :  %10.3f ms\n", cpu_ms);
    printf("    GPU time       :  %10.3f ms\n", gpu_ms);
    printf("    Speedup        :  %10.2fx\n", speedup);
    printf("    Correctness    :  %s\n", ok ? "PASS" : "FAIL");
    printf("  ----------------------------------------------------\n\n");
}

#endif // TIMER_H
