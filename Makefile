# Makefile — builds all four CUDA benchmarks.
#
# Usage:
#   make            # compiles everything
#   make run        # compiles, then runs each benchmark in order
#   make clean      # deletes the binaries
#
# If your GPU is older or newer, change ARCH below. Common values:
#   sm_60  Pascal  (GTX 10xx, P100)
#   sm_70  Volta   (V100, Titan V)
#   sm_75  Turing  (RTX 20xx, T4)
#   sm_80  Ampere  (A100)
#   sm_86  Ampere  (RTX 30xx)
#   sm_89  Ada     (RTX 40xx)
#   sm_90  Hopper  (H100)
# Or just delete the -arch flag and nvcc will pick a default.

NVCC  := nvcc
ARCH  := sm_75
FLAGS := -O3 -arch=$(ARCH) -std=c++17

BINS := vector_add matrix_mul image_blur monte_carlo_pi

all: $(BINS)

vector_add:     01_vector_add.cu     timer.h
	$(NVCC) $(FLAGS) 01_vector_add.cu     -o vector_add

matrix_mul:     02_matrix_mul.cu     timer.h
	$(NVCC) $(FLAGS) 02_matrix_mul.cu     -o matrix_mul

image_blur:     03_image_blur.cu     timer.h
	$(NVCC) $(FLAGS) 03_image_blur.cu     -o image_blur image.png

monte_carlo_pi: 04_monte_carlo_pi.cu timer.h
	$(NVCC) $(FLAGS) 04_monte_carlo_pi.cu -lcurand -o monte_carlo_pi

run: all
	@./vector_add
	@./matrix_mul 
	@./image_blur 
	@./monte_carlo_pi

clean:
	rm -f $(BINS)

.PHONY: all run clean
