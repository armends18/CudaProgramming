// =============================================================================
//  02_matrix_mul.cu
//  Benchmark: dense matrix multiplication  C = A * B   (N x N)
//
//  This is COMPUTE-BOUND, so the GPU absolutely dominates. We compare:
//    CPU :  triple nested loop (the obvious O(N^3) implementation)
//    GPU :  (a) naive global-memory kernel
//           (b) tiled shared-memory kernel — classic CUDA optimization
//
//  Build:  nvcc -O3 -arch=sm_60 02_matrix_mul.cu -o matrix_mul
//  Run  :  ./matrix_mul
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "timer.h"

#define TILE 16   // tile width for the shared-memory kernel

// ---------- CPU version: textbook triple loop ----------
void matMulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < N; ++k) acc += A[i*N + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
    }
}

// ---------- GPU kernel (naive): one thread per output element ----------
__global__ void matMulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < N; ++k) acc += A[row*N + k] * B[k*N + col];
    C[row*N + col] = acc;
}

// ---------- GPU kernel (tiled): uses shared memory to reuse loaded data ----------
//
//  Each TILE x TILE block of C is computed by one thread block. The block
//  loads a TILE x TILE tile of A and a TILE x TILE tile of B into shared
//  memory, accumulates partial dot products, slides to the next tile, repeats.
//  This cuts global-memory traffic by a factor of TILE (~16x here).
__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        // Cooperatively load one tile of A and one tile of B
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row*N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow*N + col] : 0.f;

        __syncthreads();   // wait for ALL threads in block to finish loading

        // Multiply the two tiles together; values come from FAST shared memory
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();   // wait before overwriting tiles in the next iter
    }

    if (row < N && col < N) C[row*N + col] = acc;
}

int main() {
    print_header("Benchmark 2 — Matrix Multiplication  C = A * B");

    const int N = 1024;                  // 1024 x 1024 matrices
    const size_t bytes = N * N * sizeof(float);
    printf("  Matrix size : %d x %d  (%.1f MB each)\n",
           N, N, bytes / (1024.0 * 1024.0));

    // ---------- Initialize host arrays ----------
    std::vector<float> h_A(N*N), h_B(N*N), h_C_cpu(N*N), h_C_gpu(N*N);
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = static_cast<float>((i % 13) - 6) * 0.1f;
        h_B[i] = static_cast<float>((i % 17) - 8) * 0.1f;
    }

    // ---------- CPU run ----------
    printf("  Running CPU (this is the slow part)...\n");
    CpuTimer ct; ct.start();
    matMulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    double cpu_ms = ct.stop_ms();

    // ---------- GPU setup ----------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Warm-up
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- GPU naive ----------
    GpuTimer gt; gt.start();
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    float gpu_naive_ms = gt.stop_ms();
    CUDA_CHECK(cudaGetLastError());

    // ---------- GPU tiled ----------
    GpuTimer gt2; gt2.start();
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    float gpu_tiled_ms = gt2.stop_ms();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // ---------- Verify (tolerance is generous for float ops) ----------
    bool ok = true;
    for (int i = 0; i < N*N; ++i) {
        if (std::fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-2f) { ok = false; break; }
    }

    printf("\n");
    printf("  ----------------------------------------------------\n");
    printf("    CPU                 :  %10.3f ms\n", cpu_ms);
    printf("    GPU (naive)         :  %10.3f ms   (%.1fx vs CPU)\n",
           gpu_naive_ms, cpu_ms / gpu_naive_ms);
    printf("    GPU (tiled shared)  :  %10.3f ms   (%.1fx vs CPU)\n",
           gpu_tiled_ms, cpu_ms / gpu_tiled_ms);
    printf("    Correctness         :  %s\n", ok ? "PASS" : "FAIL");
    printf("  ----------------------------------------------------\n\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
