# Makefile
CC=gcc
CFLAGS=-O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

CU=nvcc	#/usr/local/cuda-12/bin/nvcc
CUDAFLAGS=-O3 --expt-relaxed-constexpr -arch=sm_52

.PHONY: clean all

all: gengraph bfseq bfomp bfcuda

gengraph: src/gengraph.c
	@mkdir -p bin/
	@mkdir -p input/test/
	$(CC) $(CFLAGS) $< -o bin/$@.out

bfseq: src/bfseq.c
	@mkdir -p bin/
	@mkdir -p output/test
	$(CC) $(CFLAGS) $< -o bin/$@.out

bfomp: src/bfomp.c
	@mkdir -p bin/
	@mkdir -p output/omp
	$(CC) $(CFLAGS) $< -fopenmp -o bin/$@.out

bfcuda: src/bfcuda.cu
	@mkdir -p bin/
	@mkdir -p output/cuda
	$(CU) $(CUDAFLAGS) $< -o bin/$@.out

clean:
	$(RM) bin/*.out
	$(RM) -rf input/test/
	$(RM) -rf output/
	$(RM) -rf bin/
