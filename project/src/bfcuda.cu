
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define INF 1000000


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

    if(i == V-1 && has_changed) *has_negative = 1;

}

void bellmanford(int V, int *graph, int source, int *dist, int *has_negative, float *gpu_time, int blocks_grid, int block_dim){

	int *d_graph, *d_dist;
	int *d_has_changed, h_has_changed;
    int *d_has_negative, h_has_negative;

	cudaMalloc(&d_graph, sizeof(int) * V * V);
	cudaMalloc(&d_dist, sizeof(int) * V);
	cudaMalloc(&d_has_changed, sizeof(int));
    cudaMalloc(&d_has_negative, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0;

	for(int i = 0 ; i < V; i ++){
		dist[i] = INF;
	}
	dist[source] = 0;

	cudaMemcpy(d_graph, graph, sizeof(int) * V * V, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int) * V, cudaMemcpyHostToDevice);

    h_has_negative = 0;

	for(int i = 0; i < V; i++){
		h_has_changed = 0;
		cudaMemcpy(d_has_changed, &h_has_changed, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_has_negative, &h_has_negative, sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
		bellmanford_kernel<<<blocks_grid, block_dim>>>(i, V, d_graph, d_dist, d_has_changed, d_has_negative);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        *gpu_time += time;

		cudaMemcpy(&h_has_changed, d_has_changed, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(has_negative, d_has_negative, sizeof(int), cudaMemcpyHostToHost);

		if(!h_has_changed || h_has_negative){
			break;
		}

	}

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
    int blocks_grid = argv[4] ? atoi(argv[4]) : (V + block_dim- 1) / block_dim;

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