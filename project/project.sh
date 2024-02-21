# Builds and executes the commands necessary to test the execution


make clean all || exit 1

echo "                                                                                                                        "
echo "----------------------------------------Generating Graphs for Testing---------------------------------------------------" 
echo "                                                                                                                        "
# Positive directed graphs used to test omp and cuda standard parallel implementations
./bin/gengraph.out 1000 1000 0 80 "1000_pos"
./bin/gengraph.out 2000 1000 0 80 "2000_pos"
./bin/gengraph.out 3000 1000 0 80 "3000_pos"
./bin/gengraph.out 4000 1000 0 80 "4000_pos"

# Directed graphs also with negative weights to test omp and cuda negative cycle detection 
./bin/gengraph.out 2000 1000 1 80 "2000_neg"
echo "                                                                                                                        "

# Tests to the OMP implementation of Bellman-Ford, varying the number of threads and the number of vertices of the input graph
echo "------------------------------------------Bellman-Ford OMP w/ Positive Weights------------------------------------------"
echo "                                                                                                                        "
echo "Number of Threads: 1                                                                                                    "
OMP_NUM_THREADS=1 ./bin/bfomp.out 0 "1000_pos"
OMP_NUM_THREADS=1 ./bin/bfomp.out 0 "2000_pos"
OMP_NUM_THREADS=1 ./bin/bfomp.out 0 "3000_pos"
OMP_NUM_THREADS=1 ./bin/bfomp.out 0 "4000_pos"
echo "                                                                                                                        "
echo "Number of Threads: 2                                                                                                    "
OMP_NUM_THREADS=2 ./bin/bfomp.out 0 "1000_pos"
OMP_NUM_THREADS=2 ./bin/bfomp.out 0 "2000_pos"
OMP_NUM_THREADS=2 ./bin/bfomp.out 0 "3000_pos"
OMP_NUM_THREADS=2 ./bin/bfomp.out 0 "4000_pos"
echo "                                                                                                                        "
echo "Number of Threads: 4                                                                                                    "
OMP_NUM_THREADS=4 ./bin/bfomp.out 0 "1000_pos"
OMP_NUM_THREADS=4 ./bin/bfomp.out 0 "2000_pos"
OMP_NUM_THREADS=4 ./bin/bfomp.out 0 "3000_pos"
OMP_NUM_THREADS=4 ./bin/bfomp.out 0 "4000_pos"
echo "                                                                                                                        "
echo "Number of Threads: 8                                                                                                    "
OMP_NUM_THREADS=8 ./bin/bfomp.out 0 "1000_pos"
OMP_NUM_THREADS=8 ./bin/bfomp.out 0 "2000_pos"
OMP_NUM_THREADS=8 ./bin/bfomp.out 0 "3000_pos"
OMP_NUM_THREADS=8 ./bin/bfomp.out 0 "4000_pos"
echo "                                                                                                                        "

# Tests to the CUDA implementation of Bellman-Ford, varying the number of vertices of the input graph and the block dimensions
echo "------------------------------------------Bellman-Ford CUDA w/ Positive Weights-----------------------------------------"
echo "                                                                                                                        "
echo "Block Dimensions: 32                                                                                                    "
./bin/bfcuda.out 0 "1000_pos" 32
./bin/bfcuda.out 0 "2000_pos" 32
./bin/bfcuda.out 0 "3000_pos" 32
./bin/bfcuda.out 0 "4000_pos" 32
echo "                                                                                                                        "
echo "Block Dimensions: 64                                                                                                    "
./bin/bfcuda.out 0 "1000_pos" 64
./bin/bfcuda.out 0 "2000_pos" 64
./bin/bfcuda.out 0 "3000_pos" 64
./bin/bfcuda.out 0 "4000_pos" 64
echo "                                                                                                                        "
echo "Block Dimensions: 128                                                                                                    "
./bin/bfcuda.out 0 "1000_pos" 128
./bin/bfcuda.out 0 "2000_pos" 128
./bin/bfcuda.out 0 "3000_pos" 128
./bin/bfcuda.out 0 "4000_pos" 128
echo "                                                                                                                         "
echo "Block Dimensions: 256                                                                                                    "
./bin/bfcuda.out 0 "1000_pos" 256
./bin/bfcuda.out 0 "2000_pos" 256
./bin/bfcuda.out 0 "3000_pos" 256
./bin/bfcuda.out 0 "4000_pos" 256
echo "                                                                                                                        "

# Tests to the OMP implementation of Bellman-Ford, varying the number of threads of the input graph, on a graph that has negative 
# weights
echo "------------------------------------------Bellman-Ford OMP w/ Positive/Negative Weights---------------------------------"
echo "                                                                                                                        "
echo "Number of Threads: 1"
OMP_NUM_THREADS=1 ./bin/bfomp.out 0 "2000_neg"
echo "Number of Threads: 2"
OMP_NUM_THREADS=2 ./bin/bfomp.out 0 "2000_neg"
echo "Number of Threads: 4"
OMP_NUM_THREADS=4 ./bin/bfomp.out 0 "2000_neg"
echo "Number of Threads: 8"
OMP_NUM_THREADS=8 ./bin/bfomp.out 0 "2000_neg"
echo "                                                                                                                        "

# Tests to the CUDA implementation of Bellman-Ford, varying the number of vertices of the input graph and the block dimensions
echo "-----------------------------------------Bellman-Ford CUDA w/ Positive/Negative Weights---------------------------------"
echo "                                                                                                                        "
./bin/bfcuda.out 0 "2000_neg" 64
echo "                                                                                                                        "
echo "------------------------------------------------------------------------------------------------------------------------"
