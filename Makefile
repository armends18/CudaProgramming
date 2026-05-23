# Makefile — builds all four CUDA benchmarks.
#
# Usage:
#   make            # compiles everything safely
#   make run        # compiles, then runs each benchmark in sequence
#   make clean      # deletes the compiled binaries

NVCC  := nvcc
ARCH  := sm_75
FLAGS := -O3 -arch=$(ARCH) -std=c++17

BINS := vector_add matrix_mul image_blur monte_carlo_pi

all: $(BINS)

vector_add:     01_vector_add.cu     timer.h
	$(NVCC) $(FLAGS) 01_vector_add.cu     -o vector_add

matrix_mul:     02_matrix_mul.cu     timer.h
	$(NVCC) $(FLAGS) 02_matrix_mul.cu     -o matrix_mul

# FIXED: Removed 'image.png' from the compilation line. 
# We only compile source files (.cu) here!
image_blur:     03_image_blur.cu     timer.h
	$(NVCC) $(FLAGS) 03_image_blur.cu     -o image_blur 

monte_carlo_pi: 04_monte_carlo_pi.cu timer.h
	$(NVCC) $(FLAGS) 04_monte_carlo_pi.cu -lcurand -o monte_carlo_pi

# FIXED: Moved 'image.png' here so it is supplied as an argument 
# to the executable program at runtime.
run: all
	@./vector_add
	@./matrix_mul 
	@./image_blur image.png
	@./monte_carlo_pi

clean:
	rm -f $(BINS)

.PHONY: all run clean
