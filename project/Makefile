# Makefile
CC=gcc
CFLAGS=-O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

CU=nvcc	#/usr/local/cuda-12/bin/nvcc
CUDAFLAGS=-O3 --expt-relaxed-constexpr -arch=sm_52

.PHONY: clean all

all: bfomp bfcuda

%: src/%.c
	@mkdir -p bin/
	@mkdir -p input/
	@mkdir -p output/omp
	$(CC) $(CFLAGS) $< -fopenmp -o bin/$@.out

%: src/%.cu
	@mkdir -p bin/
	@mkdir -p input/
	@mkdir -p output/cuda
	$(CU) $(CUDAFLAGS) $< -o bin/$@.out

clean:
	$(RM) bin/*.out
