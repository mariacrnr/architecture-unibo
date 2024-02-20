make clean all

echo "                                                                                                                        "
echo "----------------------------------------Generating Graphs for Testing---------------------------------------------------" 
echo "                                                                                                                        "
# Positive directed graphs used to test omp and cuda standard parallel implementations
./bin/gengraph.out 1000 1000 0 50 "1000_pos.txt"
./bin/gengraph.out 2000 1000 0 50 "2000_pos.txt"
./bin/gengraph.out 3000 1000 0 50 "3000_pos.txt"
./bin/gengraph.out 4000 1000 0 50 "4000_pos.txt"

# Directed graphs also with negative weights to test omp and cuda negative cycle detection 
./bin/gengraph.out 1000 1000 1 50 "1000_neg.txt"
./bin/gengraph.out 2000 1000 1 50 "2000_neg.txt"
./bin/gengraph.out 3000 1000 1 50 "3000_neg.txt"
./bin/gengraph.out 4000 1000 1 50 "4000_neg.txt"
echo "                                                                                                                        "
echo "------------------------------------------------------------------------------------------------------------------------"