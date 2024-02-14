# Makefile
CC=gcc
CFLAGS=-O3 -Wall -Wextra -Werror -pedantic -Wconversion -Wshadow

CU=nvcc	#/usr/local/cuda-12/bin/nvcc
CUDAFLAGS=-O3 --expt-relaxed-constexpr

.PHONY: clean all

all: bfomp

%: src/%.c
	@mkdir -p bin/
	@mkdir -p input/
	@mkdir -p output/
	$(CC) $(CFLAGS) $< -fopenmp -o bin/$@.out

%: src/%.cu
	@mkdir -p bin/
	@mkdir -p input/
	@mkdir -p output/
	$(CU) $(CUDAFLAGS) $< -o bin/$@.out

clean:
	$(RM) bin/*.out
