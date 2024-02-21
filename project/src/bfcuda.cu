// bfcuda.cu
// Implements the CUDA parallel version of the Bellman-Ford Algorithm.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define INF 1000000

/**
 * Read the input graph from a file. The file should have the number of vertices on the first line followed by the adjacency matrix of the graph.
 *
 * @param filename Name of the file containing the graph.
 * @param V Pointer to store the number of vertices read from the file.
 * @return Pointer to the array representing the graph if successful, otherwise NULL.
 */
int* read_input(char* filename, int *V) {
    char folder[] = "input/test/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(strcat(path, filename), ".txt"), "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening input file.\n");
        return NULL;
    }
    
    if (fscanf(file, "%d", V) != 1) {
        fprintf(stderr, "Error reading vertex count.\n");
        fclose(file);
        return NULL;
    }

    int* graph = (int*) malloc((size_t)(*V) * (size_t)(*V) * sizeof(int *));
    if (graph == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *V; i++) {
        for (int j = 0; j < *V; j++) {
            char token[10]; 
            if (fscanf(file, "%s", token) == 1) {
                if (strcmp(token, "INF") == 0) {
                    graph[i * (*V) + j] = INF;
                } else {
                    graph[i * (*V) + j] = atoi(token);
                }
            }
        }
    }

    fclose(file);

    return graph;
}

/**
 * Write the output distances from the Bellman-Ford algorithm to a file. If the graph contains a negative cycle, 
 * it writes a message indicating the presence of a negative cycle.
 *
 * @param filename Name of the output file.
 * @param V Number of vertices in the graph.
 * @param distances Array containing the distances from the source vertex.
 * @param has_negative Flag indicating whether the graph contains a negative cycle.
 * @param blocks_grid Number of blocks in the grid.
 * @param block_dim Dimension of each block.
 */
void write_output(char* filename, int V, int *distances, int has_negative, int blocks_grid, int block_dim){
    char folder[] = "output/cuda/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    char sfilename[512];
    sprintf(sfilename, "%s_%d_%d.txt", filename, blocks_grid, block_dim);

    FILE *file = fopen(strcat(path, sfilename), "w");

    if (file == NULL) {
        fprintf(stderr, "Error opening output file.\n");
        return;
    }

    if(has_negative){
        fprintf(file, "Graph contains negative cycle!!");
    } else {
        for (int i = 0; i < V; i++) {
            fprintf(file, "%d\t\t%d\n", i, distances[i]);
        }
    }
    fprintf(file, "\n");

    fclose(file);
}

/** 
 * CUDA kernel to perform the Bellman-Ford algorithm relaxation operation. This CUDA kernel performs the Bellman-Ford algorithm on a GPU using parallel threads.
 * Each thread handles a subset of vertices to compute the shortest paths and negative cycles are detected.
 *
 * @param i Current iteration of the algorithm.
 * @param V Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param dist Array to store the distances from the source vertex.
 * @param has_changed Pointer to a flag indicating whether any distance has changed in the current iteration.
 * @param has_negative Pointer to a flag indicating whether the graph contains a negative cycle.
 */
__global__ void bellmanford_kernel(int i, int V, int *graph, int *dist, int *has_changed, int *has_negative){
    int block_index = blockDim.x * blockIdx.x + threadIdx.x;
	int block_inc = blockDim.x * gridDim.x;

	if(block_index < V){
        for(int u = 0 ; u < V ; u ++){
            for(int v = block_index; v < V; v+= block_inc){
                int updated_dist = graph[u * V + v] + dist[u];
                if (graph[u * V + v] < INF && updated_dist < dist[v]){
                    dist[v] = updated_dist;
                    *has_changed = 1;
                }
            }
        }
    }
    __syncthreads(); 

    if(i == V-1 && has_changed) *has_negative = 1; // There is a negative cycle if the distance decreases on the last iteration

}

/**
 * Implements the Bellman-Ford algorithm using CUDA on a GPU, by initializing GPU memory, launching a cuda kernel for the computation of the relax operation for each
 * vertex, and retrieves the results.
 *
 * @param V Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param source Source vertex from which shortest paths are computed.
 * @param dist Array to store the distances from the source vertex.
 * @param has_negative Pointer to a flag indicating whether the graph contains a negative cycle.
 * @param gpu_time Pointer to a variable to store the GPU execution time.
 * @param blocks_grid Number of blocks in the grid.
 * @param block_dim Dimension of each block.
 */
void bellmanford(int V, int *graph, int source, int *dist, int *has_negative, float *gpu_time, int blocks_grid, int block_dim){

	int *d_graph, *d_dist;
	int *d_has_changed, h_has_changed;
    int *d_has_negative;

    // Allocate memory for the device variables
	cudaMalloc(&d_graph, sizeof(int) * V * V);
	cudaMalloc(&d_dist, sizeof(int) * V);
	cudaMalloc(&d_has_changed, sizeof(int));
    cudaMalloc(&d_has_negative, sizeof(int));

    // Create CUDA event to gather the GPU time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0;

    // Distributes the vertices for the threads
	for(int i = 0 ; i < V; i ++){
		dist[i] = INF;
	}
	dist[source] = 0;

    // Copy the input graph array and the input distance array to the device memory for the GPU
	cudaMemcpy(d_graph, graph, sizeof(int) * V * V, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int) * V, cudaMemcpyHostToDevice);

    *has_negative = 0;

	for(int i = 0; i < V; i++){
		h_has_changed = 0;

        // Copy host memory to device memory
		cudaMemcpy(d_has_changed, &h_has_changed, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_has_negative, has_negative, sizeof(int), cudaMemcpyHostToDevice);

        // Record GPU time and run kernel on GPU
        cudaEventRecord(start);
		bellmanford_kernel<<<blocks_grid, block_dim>>>(i, V, d_graph, d_dist, d_has_changed, d_has_negative);
        cudaDeviceSynchronize(); // Syncronize threads
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        *gpu_time += time;

        // Copy device memory back to host memory
        cudaMemcpy(&h_has_changed, d_has_changed, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(has_negative, d_has_negative, sizeof(int), cudaMemcpyDeviceToHost);

        // Terminate early are no changes from the last loop
		if(!h_has_changed){ 
			break;
		}

	}

    // Only copy distance array to host if there is no negative cycle
	if(! *has_negative){
		cudaMemcpy(dist, d_dist, sizeof(int) * V, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_graph);
	cudaFree(d_dist);
	cudaFree(d_has_changed);
}

int main(int argc, char **argv){

    if (argc < 4 || argc >= 6) {
        printf("Usage: %s source_vertex filename block_dim [blocks_grid] \n", argv[0]);
        return 1;
    }

    int source = atoi(argv[1]);
    char *filename = argv[2];

    int V, has_negative;
    int *graph = read_input(filename, &V);
    if(graph == NULL) return 1;

    int block_dim = atoi(argv[3]);
    int blocks_grid = argv[4] ? atoi(argv[4]) : (V + block_dim- 1) / block_dim; // Defined dynamically by the block dimensions or specified when running

    int *dist = (int*) malloc(sizeof(int) * (size_t)V);

    float gpu_time = 0;

    cudaDeviceReset();
    bellmanford(V, graph, source, dist, &has_negative, &gpu_time, blocks_grid, block_dim);
    cudaDeviceSynchronize();

    printf("%d Vertices-> Elapsed Time: %f milliseconds\n", V, gpu_time);

    write_output(filename, V, dist, has_negative, blocks_grid, block_dim);

    free(dist);
    free(graph);

    return 0;
}