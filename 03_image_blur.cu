// =============================================================================
//  03_image_blur.cu
//  Benchmark: Real-World 2D box-blur on a user-provided image
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "timer.h"

// CRITICAL FIX: Undefine the macro conflict before pulling in the implementation
#undef stbi__err

// Define the STB Image implementation flags BEFORE including them.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RADIUS 5      // window = (2*5+1)^2 = 121 pixels averaged per output

// ---------- CPU version ----------
void blurCPU(const float* in, float* out, int W, int H) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.f;
            int cnt = 0;
            for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
                for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                        sum += in[ny*W + nx];
                        ++cnt;
                    }
                }
            }
            out[y*W + x] = sum / cnt;
        }
    }
}

// ---------- GPU kernel ----------
__global__ void blurGPU(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sum = 0.f;
    int cnt = 0;
    #pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        #pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                sum += in[ny*W + nx];
                ++cnt;
            }
        }
    }
    out[y*W + x] = sum / cnt;
}

int main(int argc, char** argv) {
    print_header("Benchmark 3 — 2D Image Blur on Custom File");

    // Check if the user passed an input file path
    if (argc < 2) {
        printf("Error: Missing input image path!\n");
        printf("Usage: %s <path_to_image.jpg_or_png>\n", argv[0]);
        return -1;
    }

    const char* input_filename = argv[1];
    int W, H, channels;

    // 1. LOAD THE IMAGE & FORCE TO GRAYSCALE (1 Channel)
    printf("  Loading input image: %s...\n", input_filename);
    unsigned char* img_data = stbi_load(input_filename, &W, &H, &channels, 1);

    if (!img_data) {
        printf("Error: Failed to load image file '%s'.\n", input_filename);
        return -1;
    }

    const size_t bytes = W * H * sizeof(float);
    printf("  Image Resolution: %d x %d   (%.1f MB raw float size)\n", W, H, bytes / (1024.0 * 1024.0));
    printf("  Filter Window   : %d x %d box blur\n", 2*RADIUS+1, 2*RADIUS+1);

    // 2. CONVERT INTENSITY DATA FROM UC (0-255) TO FLOAT (0.0-1.0)
    std::vector<float> h_in(W*H), h_out_cpu(W*H), h_out_gpu(W*H);
    for (int i = 0; i < W * H; ++i) {
        h_in[i] = static_cast<float>(img_data[i]) / 255.0f;
    }
    stbi_image_free(img_data); // Clean up the raw disk bytes immediately

    // ---------- CPU run ----------
    printf("  Running CPU baseline...\n");
    CpuTimer ct; ct.start();
    blurCPU(h_in.data(), h_out_cpu.data(), W, H);
    double cpu_ms = ct.stop_ms();

    // ---------- GPU run ----------
    printf("  Running GPU acceleration...\n");
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Define 2D thread configuration mapping tiles over the image dimensions
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    // Warm-up step
    blurGPU<<<grid, block>>>(d_in, d_out, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer gt; gt.start();
    blurGPU<<<grid, block>>>(d_in, d_out, W, H);
    float gpu_ms = gt.stop_ms();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // ---------- Verify Math Accuracies ----------
    bool ok = true;
    for (int i = 0; i < W*H; ++i) {
        if (std::fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-4f) { ok = false; break; }
    }

    print_results(cpu_ms, gpu_ms, ok);

    // 3. CONVERT BACK TO UNSIGNED CHARS AND WRITE FILE BACK TO DISK
    std::vector<unsigned char> out_bytes(W * H);
    for (int i = 0; i < W * H; ++i) {
        float val = h_out_gpu[i] * 255.0f;
        out_bytes[i] = static_cast<unsigned char>(fmaxf(0.0f, fminf(val, 255.0f)));
    }

    const char* output_filename = "blurred_output.png";
    printf("  Writing output image to disk: %s...\n", output_filename);
    stbi_write_png(output_filename, W, H, 1, out_bytes.data(), W);

    // ---------- Resource Cleanup ----------
    cudaFree(d_in);
    cudaFree(d_out);
    printf("  Process complete!\n");
    return 0;
}
