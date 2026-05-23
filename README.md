# CUDA CPU vs GPU Benchmark Suite

Four side-by-side comparisons of CPU and GPU implementations of the same task. Each `.cu` file contains both the sequential CPU version and the parallel CUDA kernel, runs both, verifies they produce the same answer, and prints timing and speedup.

## What's included

| # | File | Task | Why it matters |
|---|------|------|----------------|
| 1 | `01_vector_add.cu`     | `C[i] = A[i] + B[i]` on 64M floats     | Memory-bound — a modest but real speedup |
| 2 | `02_matrix_mul.cu`     | `C = A * B`, 1024 x 1024 matrices      | Compute-bound — huge speedup; includes a tiled shared-memory kernel |
| 3 | `03_image_blur.cu`     | 11 x 11 box blur on a 4K x 4K image    | 2D stencil — shows 2D thread grids |
| 4 | `04_monte_carlo_pi.cu` | Monte Carlo estimation of pi           | Embarrassingly parallel — uses cuRAND on-device |

`timer.h` is a shared header with CPU/GPU timing helpers, CUDA error checking, and a results-printing utility.

## Requirements

- An NVIDIA GPU with CUDA capability 6.0 or newer (Pascal+).
- The CUDA Toolkit installed (`nvcc` on your PATH).
- A C++17-capable host compiler (g++, clang, MSVC).

No external libraries beyond what ships with the CUDA Toolkit. The Monte Carlo benchmark uses cuRAND, which is included.

## Build & run

```bash
make            # compile all four
make run        # compile and run all four
make clean      # delete binaries
```

Or compile individually:

```bash
nvcc -O3 -arch=sm_60 01_vector_add.cu     -o vector_add
nvcc -O3 -arch=sm_60 02_matrix_mul.cu     -o matrix_mul
nvcc -O3 -arch=sm_60 03_image_blur.cu     -o image_blur
nvcc -O3 -arch=sm_60 04_monte_carlo_pi.cu -lcurand -o monte_carlo_pi
```

**Change `-arch=sm_XX`** in the Makefile to match your GPU (see the comment block inside the Makefile for the lookup table). If you remove the flag entirely, `nvcc` picks a default — usually fine.

### Running on Google Colab

If you don't have an NVIDIA GPU locally, Google Colab gives you a free one:

1. Open a new notebook, **Runtime → Change runtime type → T4 GPU**.
2. Upload all five files (the four `.cu` files, `timer.h`, and `Makefile`).
3. In a cell: `!nvcc --version` (verify the toolkit is there).
4. In a cell: `!make run`.

## What to expect

Numbers vary wildly by hardware. On a modern desktop CPU vs a mid-range GPU (e.g. RTX 3060 / T4) you'll typically see something like:

| Benchmark           | CPU time      | GPU time     | Speedup        |
|---------------------|---------------|--------------|----------------|
| Vector addition     | ~200 ms       | ~10 ms       | 10–30x         |
| Matrix mul (naive)  | ~5–10 s       | ~30 ms       | 100–300x       |
| Matrix mul (tiled)  | (same CPU)    | ~10 ms       | 500–1000x      |
| Image blur          | ~10 s         | ~30 ms       | 200–400x       |
| Monte Carlo pi      | ~2 s / 100M   | ~50 ms / 10G | 1000x+ per-dart|

If your numbers are off by an order of magnitude in either direction, that's normal — hardware varies hugely, and CPU compilers do a *lot* with `-O3`.

## How the timing works

- **CPU timing** uses `std::chrono::high_resolution_clock`.
- **GPU timing** uses CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`), which measure actual GPU execution time, not host-side wait time.
- We do a **warm-up launch** before the timed launch on the GPU. The first kernel launch in a process pays one-time JIT and context-init costs — including those would make the GPU look artificially slow.
- GPU timing **does not** include host-to-device or device-to-host memory transfers. For these algorithms the kernel time is what we want to compare; if you care about end-to-end wall time (data has to move over PCIe), wrap the `cudaMemcpy` calls in the timer too.

## Why each speedup looks the way it does

- **Vector add** is memory-bound. Both CPU and GPU just stream data through; the GPU has more memory bandwidth but it's not 100x more. Expect 10–30x.
- **Matrix multiply** is compute-bound: O(N³) work for O(N²) data, so the same numbers get reused many times. The GPU's massive ALU count crushes it. The tiled version reuses data through shared memory and gets a further few-x improvement.
- **Image blur** is also compute-heavy per pixel (121 reads + sum) and 2D-friendly. Big GPU win.
- **Monte Carlo** is the textbook embarrassingly-parallel case — every dart is independent, no synchronization between threads except one atomic at the end.

## Modifying the code

Each file has `const int N = ...` (or `W, H`, etc.) near the top of `main()`. Bump it up if your GPU is fast and you want bigger numbers, or down if you're on Colab and don't want to wait.

The matrix-mul `CPU` version is the **slow** part — it's an unoptimized triple loop. If you want a more honest CPU baseline, link against OpenBLAS or MKL and use `cblas_sgemm`. That'll narrow the gap a lot, but the GPU still wins.

## License

Public domain — use it however you like for your coursework.
